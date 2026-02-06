import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py
from typing import List, Optional


def load_data(path):
    """Load data stored in npz or h5 format."""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data_file:
            if "gns_data" in data_file:
                data = data_file["gns_data"]
            else:
                data = [item for _, item in data_file.items()]
    elif path.endswith(".h5"):
        with h5py.File(path, "r") as data_file:
            data = []
            for key in data_file.keys():
                trajectory = data_file[key]
                positions = trajectory["positions"][()]
                particle_type = trajectory["particle_type"][()]
                material_property = trajectory["material_property"][()]
                data.append((positions, particle_type, material_property))
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5 files.")
    return data


class ParticleDataset(Dataset):
    def __init__(
        self,
        file_path,
        input_sequence_length=6,
        mode="sample",
        rigid_body_types: Optional[List[int]] = None,
    ):
        self.file_path = file_path
        self.input_sequence_length = input_sequence_length
        self.mode = mode
        self.rigid_body_types = (
            sorted(set(int(t) for t in rigid_body_types))
            if rigid_body_types is not None
            else []
        )
        self.include_rigid_bodies = len(self.rigid_body_types) > 0
        self.data = load_data(file_path)
        self._preprocess_data()

    @staticmethod
    def _expand_particle_feature(value, n_particles, dtype):
        """Return a per-particle 1D array, supporting scalar or per-particle input."""
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            return np.full(n_particles, arr.item(), dtype=dtype)

        arr = arr.reshape(-1)
        if arr.size == 1:
            return np.full(n_particles, arr.item(), dtype=dtype)
        if arr.size == n_particles:
            return arr.astype(dtype, copy=False)
        raise ValueError(
            f"Expected scalar or length-{n_particles} feature array, got shape {arr.shape}"
        )

    def _build_rigid_bodies(self, particle_type: np.ndarray) -> List[torch.Tensor]:
        """
        Build list of rigid body index tensors for this sample/trajectory.
        Each particle type in `self.rigid_body_types` defines one rigid body.
        """
        rigid_bodies = []
        for rigid_type in self.rigid_body_types:
            rigid_indices = np.where(particle_type == rigid_type)[0]
            if rigid_indices.size < 2:
                continue
            rigid_bodies.append(torch.from_numpy(rigid_indices.astype(np.int64)))
        return rigid_bodies

    def _preprocess_data(self):
        self.dimension = self.data[0][0].shape[-1]
        self.material_property_as_feature = len(self.data[0]) >= 3

        if self.mode == "sample":
            self.data_lengths = [
                x.shape[0] - self.input_sequence_length for x, *_ in self.data
            ]
            self.length = sum(self.data_lengths)
            self.cumulative_lengths = np.cumsum([0] + self.data_lengths)
        else:  # trajectory mode
            self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == "sample":
            return self._get_sample(idx)
        else:  # trajectory mode
            return self._get_trajectory(idx)

    def _get_sample(self, idx):
        trajectory_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        time_idx = (
            idx - self.cumulative_lengths[trajectory_idx] + self.input_sequence_length
        )

        positions = self.data[trajectory_idx][0][
            time_idx - self.input_sequence_length : time_idx
        ]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = self._expand_particle_feature(
            self.data[trajectory_idx][1], positions.shape[0], dtype=int
        )

        n_particles_per_example = positions.shape[0]
        rigid_bodies = (
            self._build_rigid_bodies(particle_type) if self.include_rigid_bodies else None
        )

        if self.material_property_as_feature:
            material_property = self._expand_particle_feature(
                self.data[trajectory_idx][2], positions.shape[0], dtype=float
            )
            features = (
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
            )
        else:
            features = (positions, particle_type, n_particles_per_example)
        if rigid_bodies is not None:
            features = features + (rigid_bodies,)

        label = self.data[trajectory_idx][0][time_idx]

        return features, label

    def _get_trajectory(self, idx):
        if self.material_property_as_feature:
            positions, particle_type, material_property = self.data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = self._expand_particle_feature(
                particle_type, positions.shape[0], dtype=int
            )
            material_property = self._expand_particle_feature(
                material_property, positions.shape[0], dtype=float
            )
            n_particles_per_example = positions.shape[0]
            rigid_bodies = (
                self._build_rigid_bodies(particle_type)
                if self.include_rigid_bodies
                else None
            )

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                torch.tensor(material_property).to(torch.float32).contiguous(),
                n_particles_per_example,
            )
        else:
            positions, particle_type = self.data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = self._expand_particle_feature(
                particle_type, positions.shape[0], dtype=int
            )
            n_particles_per_example = positions.shape[0]
            rigid_bodies = (
                self._build_rigid_bodies(particle_type)
                if self.include_rigid_bodies
                else None
            )

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                n_particles_per_example,
            )

        if rigid_bodies is not None:
            trajectory = trajectory + (rigid_bodies,)

        return trajectory

    def get_num_features(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: The number of features.
        """
        return len(self.data[0])


def collate_fn_sample(batch):
    features, labels = zip(*batch)

    position_list = []
    particle_type_list = []
    material_property_list = []
    n_particles_per_example_list = []
    rigid_bodies_list = []
    has_any_rigid_metadata = False
    offset = 0

    for feature in features:
        position_list.append(feature[0])
        particle_type_list.append(feature[1])
        has_rigid_bodies = isinstance(feature[-1], list)
        has_any_rigid_metadata = has_any_rigid_metadata or has_rigid_bodies
        base_feature_len = len(feature) - (1 if has_rigid_bodies else 0)

        if base_feature_len == 4:  # If material property is present
            material_property_list.append(feature[2])
            n_particles_per_example_list.append(feature[3])
        else:
            n_particles_per_example_list.append(feature[2])
        if has_rigid_bodies:
            rigid_bodies = feature[-1]
            for body_indices in rigid_bodies:
                rigid_bodies_list.append(body_indices + offset)
        offset += feature[0].shape[0]

    collated_features = (
        torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
        torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        torch.tensor(n_particles_per_example_list).contiguous(),
    )

    if material_property_list:
        material_property_tensor = (
            torch.tensor(np.concatenate(material_property_list))
            .to(torch.float32)
            .contiguous()
        )
        collated_features = (
            collated_features[:2] + (material_property_tensor,) + collated_features[2:]
        )
    if has_any_rigid_metadata:
        collated_features = collated_features + (rigid_bodies_list,)

    collated_labels = torch.tensor(np.vstack(labels)).to(torch.float32).contiguous()

    return collated_features, collated_labels


def collate_fn_trajectory(batch):
    return batch  # No need for collation as each item is already a full trajectory


def get_data_loader(
    file_path,
    mode="sample",
    input_sequence_length=6,
    batch_size=32,
    shuffle=True,
    use_dist=False,
    rigid_body_types=None,
):
    """
    Get a data loader for the ParticleDataset.

    Args:
        file_path (str): Path to the data file.
        mode (str): 'sample' or 'trajectory' mode.
        input_sequence_length (int): Length of input sequence.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the data.
        use_dist (bool): Whether to use DistributedSampler for distributed training.
        rigid_body_types (list[int] | None): Particle types that each define one rigid body.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset = ParticleDataset(
        file_path,
        input_sequence_length,
        mode,
        rigid_body_types,
    )

    if use_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    else:
        sampler = None

    if mode == "sample":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn_sample,
            pin_memory=True,
        )
    else:  # trajectory mode
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_trajectory,
            pin_memory=True,
        )
