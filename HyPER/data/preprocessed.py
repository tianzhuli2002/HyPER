import os
import sys
import math
import h5py
import pickle
import base64
import json
import hashlib
import warnings
import time
from copy import deepcopy
from typing import Optional

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Dataset

from .dataset import HyPERDataset


DB_FORMAT = "pickle_vlen_uint8_v1"


def _decode_graph_payload(payload):
    """Decode new raw-pickle vlen uint8 payloads and legacy base64 strings."""
    if isinstance(payload, np.ndarray):
        return pickle.loads(payload.tobytes())
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(payload, bytes):
        try:
            return pickle.loads(payload)
        except Exception:
            return pickle.loads(base64.b64decode(payload), encoding="utf-8")
    raise TypeError(f"Unsupported graph payload type: {type(payload)}")


class PreprocessingWrite(IterableDataset):
    """Preprocess HyPER graphs and write them into an HDF5 graph database.

    Args:
        root: dataset root directory (HyPER run folder)
        name: dataset name (h5 basename without extension)
    """
    def __init__(
        self,
        root: str,
        name: str,
        training: bool = True,
        config: Optional[dict] = None,
        preprocess_chunk_size: Optional[int] = None,
    ):
        super(PreprocessingWrite).__init__()
        self.root = root
        self.name = name
        self.training = training
        self.config = deepcopy(config) if config is not None else None
        if preprocess_chunk_size is None:
            preprocess_chunk_size = int(os.getenv("HYPER_PREPROCESS_CHUNK_SIZE", "1024"))
        self.preprocess_chunk_size = max(1, int(preprocess_chunk_size))
        self.debug = os.getenv("HYPER_DB_DEBUG", "0") == "1"

        self.master = HyPERDataset(root=root, name=name, training=training, force_reload=False, config=self.config)
        self.size = len(self.master)

    @staticmethod
    def worker_bounds(size: int, num_workers: int, worker_id: int) -> tuple[int, int]:
        per_worker = int(math.ceil(size / float(num_workers)))
        iter_start = min(worker_id * per_worker, size)
        iter_end = min(iter_start + per_worker, size)
        return iter_start, iter_end

    def create_db(self, db_path: str, num_events: int):
        f = h5py.File(db_path, 'a')
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        db = f.create_dataset('Graphs', shape=(num_events,), dtype=vlen_uint8)
        return f, db

    def write_graph(
        self,
        db,
        index: int,
        G: Data,
        worker_id: int = 0,
        source_idx: Optional[int] = None,
        graph_time: Optional[float] = None,
    ) -> tuple[float, float, int]:
        t0 = time.time()
        graph_bytes = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)
        payload = np.frombuffer(graph_bytes, dtype=np.uint8)
        t_pickle = time.time() - t0
        t1 = time.time()
        db[index] = payload
        t_write = time.time() - t1
        if self.debug and index % 1000 == 0:
            graph_time = 0.0 if graph_time is None else graph_time
            print(
                f"[HYPER_DB] worker={worker_id} idx={source_idx if source_idx is not None else 'n/a'} "
                f"local={index} graph={graph_time:.4f}s pickle={t_pickle:.4f}s write={t_write:.4f}s "
                f"bytes={len(graph_bytes)}",
                file=sys.stderr,
                flush=True,
            )
        return t_pickle, t_write, len(graph_bytes)

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = self.size
            worker_id = 0
            num_workers = 1
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_start, iter_end = self.worker_bounds(self.size, num_workers, worker_id)

        # create per-worker DB file
        base = os.path.join(self.root, self.name)
        if num_workers <= 1:
            db_path = base + '-0.db'
        else:
            db_path = base + f'-{worker_id}.db'

        f, db = self.create_db(db_path, int(iter_end - iter_start))

        local_index = 0
        total_graph_time = 0.0
        total_pickle_time = 0.0
        total_write_time = 0.0
        total_bytes = 0
        total_edges = 0
        total_hyperedges = 0
        processed = 0
        wall_start = time.time()
        progress = tqdm(range(iter_start, iter_end),
                        desc=f"Worker {worker_id if worker_info else 'main'}",
                        total=iter_end-iter_start,
                        position=worker_id if worker_info is not None else 0,
                        dynamic_ncols=False, ncols=100, nrows=5, file=sys.stderr,
                        leave=True, unit='evt', miniters=100, ascii=True)
        chunk_size = self.preprocess_chunk_size
        for chunk_start in range(iter_start, iter_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, iter_end)
            t0 = time.time()
            graphs = self.master.processing_chunk(chunk_start, chunk_end)
            t_graph = time.time() - t0
            total_graph_time += t_graph
            graph_time = t_graph / max(1, len(graphs))
            for offset, G in enumerate(graphs):
                idx = chunk_start + offset
                t_pickle, t_write, nbytes = self.write_graph(
                    db,
                    local_index,
                    G,
                    worker_id=worker_id,
                    source_idx=idx,
                    graph_time=graph_time,
                )
                total_pickle_time += t_pickle
                total_write_time += t_write
                total_bytes += nbytes
                total_edges += int(G.edge_index.shape[1]) if hasattr(G, "edge_index") else 0
                total_hyperedges += int(G.hyperedge_index.shape[1]) if hasattr(G, "hyperedge_index") else 0
                processed += 1
                local_index += 1
                progress.update(1)
                yield 1
        progress.close()

        f.close()
        wall = time.time() - wall_start
        if processed > 0:
            print(
                "[HYPER_DB_PROFILE] "
                f"worker={worker_id} events={processed} total_time={wall:.3f}s "
                f"events_per_s={processed / max(wall, 1e-9):.2f} "
                f"avg_graph_s={total_graph_time / processed:.6f} "
                f"avg_pickle_s={total_pickle_time / processed:.6f} "
                f"avg_write_s={total_write_time / processed:.6f} "
                f"avg_edges={total_edges / processed:.1f} "
                f"avg_hyperedges={total_hyperedges / processed:.1f} "
                f"avg_payload_bytes={total_bytes / processed:.1f} "
                f"target_encoding={getattr(self.master, 'target_encoding', 'binary')} "
                f"vectorized_chunks={getattr(self.master, 'use_vectorized_chunks', False)} "
                f"chunk_size={chunk_size} "
                f"rebuilt=True db_part={db_path}",
                file=sys.stderr,
                flush=True,
            )


class GraphDB():
    def __init__(
        self,
        root: str,
        name: str,
        training: bool = True,
        num_workers: int = 0,
        batch_size: int = 128,
        force_reload: bool = False,
        config: Optional[dict] = None,
        preprocess_chunk_size: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        if preprocess_chunk_size is None:
            preprocess_chunk_size = int(os.getenv("HYPER_PREPROCESS_CHUNK_SIZE", "1024"))
        self.preprocess_chunk_size = max(1, int(preprocess_chunk_size))
        self.dataset_vars = None
        self._db_file = None
        self._db = None
        self._db_pid = None

        self.db_path = os.path.join(root, f"{name}.db")
        self.manifest_path = self.db_path + ".manifest.json"
        self.lock_path = self.db_path + ".build.lock"

        self.root = root
        self.name = name
        self.config = deepcopy(config) if config is not None else None
        self.config_hash = self._config_hash(self.config)

        if force_reload and os.path.exists(self.db_path):
            os.remove(self.db_path)
        if force_reload and os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

        if os.path.exists(self.db_path) and not self._is_valid_db(self.db_path):
            os.remove(self.db_path)
            if os.path.exists(self.manifest_path):
                os.remove(self.manifest_path)

        if os.path.exists(self.db_path):
            self._validate_manifest()

        self._log_cache_context()

        if not os.path.exists(self.db_path):
            self._guard_slurm_cache_rank()
            lock_fd = self._acquire_build_lock()
            print(f"Running HyPER graph pre-processing on dataset '{name}' with {num_workers} workers:")
            try:
                self.batch_pw = PreprocessingWrite(
                    root=root,
                    name=name,
                    training=training,
                    config=self.config,
                    preprocess_chunk_size=self.preprocess_chunk_size,
                )
                self.size = self.batch_pw.size
                self.dataset_vars = vars(self.batch_pw.master)
                self.batch_files = [os.path.join(root, f"{name}-{i}.db") for i in range(max(1, num_workers))]
                self.cleanup()
                self.preprocessing_write()
                self.concatenate()
                self.cleanup()
                self._write_manifest()
                print(f"Graph database {self.db_path} created.")
            finally:
                self._release_build_lock(lock_fd)
        else:
            print(f"Using existing HyPER graph database {self.db_path}.")
            master = HyPERDataset(root=root, name=name, training=training, force_reload=False, config=self.config)
            self.dataset_vars = vars(master)
            print(
                "[HYPER_DB_PROFILE] "
                f"events=reused total_time=0.000s events_per_s=inf "
                f"target_encoding={getattr(master, 'target_encoding', 'binary')} "
                f"vectorized_chunks={getattr(master, 'use_vectorized_chunks', False)} "
                f"rebuilt=False db_path={self.db_path}",
                file=sys.stderr,
                flush=True,
            )

        with h5py.File(self.db_path, 'r') as db_file:
            self.size = len(db_file['Graphs'])

    def preprocessing_write(self):
        # persistent_workers only valid when num_workers>0
        kwargs = dict(num_workers=self.num_workers, batch_size=None, shuffle=False, drop_last=False)
        if self.num_workers > 0:
            kwargs['persistent_workers'] = True
        loader = TorchDataLoader(self.batch_pw, **kwargs)
        for _ in loader:
            pass

    def concatenate(self):
        current_index = 0
        with h5py.File(self.db_path, 'a') as output_f:
            vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
            output_db = output_f.create_dataset('Graphs', shape=(self.size,), dtype=vlen_uint8)

            for input_file in self.batch_files:
                if not os.path.exists(input_file):
                    continue
                with h5py.File(input_file, 'r') as input_f:
                    input_db = input_f['Graphs']
                    num_events = len(input_db)
                    try:
                        output_db[current_index:current_index + num_events] = input_db[:]
                    except (TypeError, ValueError):
                        for offset in range(num_events):
                            output_db[current_index + offset] = input_db[offset]
                    current_index += num_events

    def cleanup(self):
        for file in self.batch_files:
            if os.path.exists(file):
                os.remove(file)

    def _log_cache_context(self) -> None:
        try:
            import socket
            host = socket.gethostname()
        except Exception:
            host = "unknown"
        print("================================")
        print("HyPER graph DB context")
        print(f"hostname: {host}")
        print(f"pid: {os.getpid()}")
        print(f"SLURM_NTASKS: {os.getenv('SLURM_NTASKS', 'unset')}")
        print(f"SLURM_PROCID: {os.getenv('SLURM_PROCID', 'unset')}")
        print(f"SLURM_LOCALID: {os.getenv('SLURM_LOCALID', 'unset')}")
        print(f"requested num_workers: {self.num_workers}")
        print(f"actual preprocessing DataLoader num_workers: {self.num_workers}")
        print(f"DB output path: {self.db_path}")
        print("================================")

    def _slurm_procid(self) -> int | None:
        value = os.getenv("SLURM_PROCID")
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _guard_slurm_cache_rank(self) -> None:
        procid = self._slurm_procid()
        if procid is not None and procid != 0:
            print(
                "HyPER graph DB build requested from non-zero Slurm rank "
                f"SLURM_PROCID={procid}. Exiting this rank to avoid concurrent DB writes.",
                flush=True,
            )
            sys.exit(0)

    def _acquire_build_lock(self) -> int:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            procid = self._slurm_procid()
            if procid is not None and procid != 0:
                print(
                    f"Build lock already exists at {self.lock_path}; non-zero Slurm rank exits.",
                    flush=True,
                )
                sys.exit(0)
            raise RuntimeError(
                f"Graph DB build lock already exists: {self.lock_path}. "
                "Another process may be building this DB. Remove the lock only if no build is running."
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "pid": os.getpid(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_ntasks": os.getenv("SLURM_NTASKS"),
                "slurm_procid": os.getenv("SLURM_PROCID"),
                "time": time.time(),
            }, sort_keys=True))
            handle.write("\n")
        return -1

    def _release_build_lock(self, _lock_fd: int) -> None:
        if os.path.exists(self.lock_path):
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

    @staticmethod
    def _is_valid_db(db_path: str) -> bool:
        try:
            with h5py.File(db_path, 'r') as db_file:
                return 'Graphs' in db_file
        except OSError:
            return False

    @staticmethod
    def _config_hash(config: Optional[dict]) -> str:
        payload = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_manifest(self) -> None:
        manifest = {
            "name": self.name,
            "config_hash": self.config_hash,
            "db_format": DB_FORMAT,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def _validate_manifest(self) -> None:
        if not os.path.exists(self.manifest_path):
            warnings.warn(
                f"Using existing graph DB without a manifest: {self.db_path}. "
                "If graph-construction config changed since it was built, rebuild with "
                "`dataset.force_reload=true`.",
                UserWarning,
            )
            return

        with open(self.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        stored_hash = manifest.get("config_hash")
        if stored_hash != self.config_hash:
            raise RuntimeError(
                f"Existing graph DB manifest does not match the current graph config: {self.db_path}. "
                "Rebuild intentionally with `dataset.force_reload=true` or choose a fresh dataset/cache name."
            )
        stored_format = manifest.get("db_format")
        if stored_format is None:
            warnings.warn(
                f"Using existing graph DB manifest without a db_format field: {self.db_path}. "
                "Old base64 graph DBs remain readable, but new DBs use raw pickle vlen uint8.",
                UserWarning,
            )
        elif stored_format != DB_FORMAT:
            warnings.warn(
                f"Using graph DB with db_format={stored_format!r}; expected {DB_FORMAT!r}. "
                "Attempting backwards-compatible graph payload decoding.",
                UserWarning,
            )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return _decode_graph_payload(self.db[index])

    @property
    def db(self):
        pid = os.getpid()
        if self._db is None or self._db_pid != pid:
            self.close()
            self._db_file = h5py.File(self.db_path, 'r')
            self._db = self._db_file['Graphs']
            self._db_pid = pid
        return self._db

    def close(self) -> None:
        if self._db_file is not None:
            try:
                self._db_file.close()
            except Exception:
                pass
        self._db_file = None
        self._db = None
        self._db_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_db_file'] = None
        state['_db'] = None
        state['_db_pid'] = None
        return state

    def __del__(self):
        self.close()


class HyPEROnDiskDataset(Dataset):
    def __init__(
        self,
        root: str,
        name: str,
        training: bool = True,
        force_reload: bool = False,
        batch_size: int = 128,
        num_workers: int = 0,
        config: Optional[dict] = None,
        preprocess_chunk_size: Optional[int] = None,
    ) -> None:
        self.root = root
        self.name = name
        self.force_reload = force_reload
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_mode = training
        self.config = deepcopy(config) if config is not None else None
        self.preprocess_chunk_size = preprocess_chunk_size

        self.db = GraphDB(
            root=root,
            name=name,
            training=training,
            num_workers=num_workers,
            batch_size=batch_size,
            force_reload=force_reload,
            config=self.config,
            preprocess_chunk_size=preprocess_chunk_size,
        )

        for key, value in self.db.dataset_vars.items():
            setattr(self, key, value)

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

    def __getitem__(self, index):
        return self.db[index]

    def __len__(self):
        return self.db.__len__()
