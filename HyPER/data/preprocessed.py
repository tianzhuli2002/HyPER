import os
import sys
import math
import h5py
import pickle
import base64
from copy import deepcopy
from typing import Optional

from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .dataset import HyPERDataset


class PreprocessingWrite(IterableDataset):
    """Preprocess HyPER graphs and write them into an HDF5 graph database.

    Args:
        root: dataset root directory (HyPER run folder)
        name: dataset name (h5 basename without extension)
    """
    def __init__(self, root: str, name: str, training: bool = True, config: Optional[dict] = None):
        super(PreprocessingWrite).__init__()
        self.root = root
        self.name = name
        self.training = training
        self.config = deepcopy(config) if config is not None else None

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
        db = f.create_dataset('Graphs', shape=(num_events,), dtype=h5py.string_dtype(encoding='utf-8'))
        return f, db

    def write_graph(self, db, index: int, G: Data) -> None:
        graph_bytes = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)
        db[index] = base64.b64encode(graph_bytes)

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

        for idx in tqdm(range(iter_start, iter_end),
                        desc=f"Worker {worker_id if worker_info else 'main'}",
                        total=iter_end-iter_start,
                        position=worker_id if worker_info is not None else 0,
                        dynamic_ncols=False, ncols=100, nrows=5, file=sys.stderr,
                        leave=True, unit='evt', miniters=100, ascii=True):
            G = self.master[idx]
            self.write_graph(db, idx - iter_start, G)
            yield G

        f.close()


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
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_vars = None
        self._db_file = None
        self._db = None
        self._db_pid = None

        self.db_path = os.path.join(root, f"{name}.db")

        self.root = root
        self.name = name
        self.config = deepcopy(config) if config is not None else None

        if force_reload and os.path.exists(self.db_path):
            os.remove(self.db_path)

        if os.path.exists(self.db_path) and not self._is_valid_db(self.db_path):
            os.remove(self.db_path)

        if not os.path.exists(self.db_path):
            print(f"Running HyPER graph pre-processing on dataset '{name}' with {num_workers} workers:")
            self.batch_pw = PreprocessingWrite(root=root, name=name, training=training, config=self.config)
            self.size = self.batch_pw.size
            self.dataset_vars = vars(self.batch_pw.master)
            self.batch_files = [os.path.join(root, f"{name}-{i}.db") for i in range(max(1, num_workers))]
            self.cleanup()
            self.preprocessing_write()
            self.concatenate()
            self.cleanup()
            print(f"Graph database {self.db_path} created.")
        else:
            master = HyPERDataset(root=root, name=name, training=training, force_reload=False, config=self.config)
            self.dataset_vars = vars(master)

        with h5py.File(self.db_path, 'r') as db_file:
            self.size = len(db_file['Graphs'])

    def preprocessing_write(self):
        # persistent_workers only valid when num_workers>0
        kwargs = dict(num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, drop_last=False)
        if self.num_workers > 0:
            kwargs['persistent_workers'] = True
        loader = DataLoader(self.batch_pw, **kwargs)
        for _ in loader:
            pass

    def concatenate(self):
        current_index = 0
        with h5py.File(self.db_path, 'a') as output_f:
            output_db = output_f.create_dataset('Graphs', shape=(self.size,), dtype=h5py.string_dtype(encoding='utf-8'))

            for input_file in self.batch_files:
                if not os.path.exists(input_file):
                    continue
                with h5py.File(input_file, 'r') as input_f:
                    input_db = input_f['Graphs']
                    num_events = len(input_db)
                    output_db[current_index:current_index + num_events] = input_db[:]
                    current_index += num_events

    def cleanup(self):
        for file in self.batch_files:
            if os.path.exists(file):
                os.remove(file)

    @staticmethod
    def _is_valid_db(db_path: str) -> bool:
        try:
            with h5py.File(db_path, 'r') as db_file:
                return 'Graphs' in db_file
        except OSError:
            return False

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        graph_bytes = base64.b64decode(self.db[index])
        return pickle.loads(graph_bytes, encoding='utf-8')

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
    ) -> None:
        self.root = root
        self.name = name
        self.force_reload = force_reload
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_mode = training
        self.config = deepcopy(config) if config is not None else None

        self.db = GraphDB(
            root=root,
            name=name,
            training=training,
            num_workers=num_workers,
            batch_size=batch_size,
            force_reload=force_reload,
            config=self.config,
        )

        for key, value in self.db.dataset_vars.items():
            setattr(self, key, value)

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

    def __getitem__(self, index):
        return self.db[index]

    def __len__(self):
        return self.db.__len__()
