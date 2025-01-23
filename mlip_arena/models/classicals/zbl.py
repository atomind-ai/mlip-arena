import torch
import torch.linalg as LA
import torch.nn as nn
import torch_scatter
from torch_geometric.data import Data

from ase.data import covalent_radii
from ase.units import _e, _eps0, m, pi
from e3nn.util.jit import compile_mode # TODO: e3nn allows autograd in compiled model


@compile_mode("script")
class ZBL(nn.Module):
    """Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion"""

    def __init__(
        self,
        trianable: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self, **kwargs)
        
        torch.set_default_dtype(torch.double)

        self.a = torch.nn.parameter.Parameter(
            torch.tensor(
                [0.18175, 0.50986, 0.28022, 0.02817], dtype=torch.get_default_dtype()
            ),
            requires_grad=trianable,
        )
        self.b = torch.nn.parameter.Parameter(
            torch.tensor(
                [-3.19980, -0.94229, -0.40290, -0.20162],
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=trianable,
        )

        self.a0 = torch.nn.parameter.Parameter(
            torch.tensor(0.46850, dtype=torch.get_default_dtype()),
            requires_grad=trianable,
        )

        self.p = torch.nn.parameter.Parameter(
            torch.tensor(0.23, dtype=torch.get_default_dtype()), requires_grad=trianable
        )

        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def phi(self, x):
        return torch.einsum("i,ij->j", self.a, torch.exp(torch.outer(self.b, x)))

    def d_phi(self, x):
        return torch.einsum(
            "i,ij->j", self.a * self.b, torch.exp(torch.outer(self.b, x))
        )

    def dd_phi(self, x):
        return torch.einsum(
            "i,ij->j", self.a * self.b**2, torch.exp(torch.outer(self.b, x))
        )

    def eij(
        self, zi: torch.Tensor, zj: torch.Tensor, rij: torch.Tensor
    ) -> torch.Tensor:  # [eV]
        return _e * m / (4 * pi * _eps0) * torch.div(torch.mul(zi, zj), rij)

    def d_eij(
        self, zi: torch.Tensor, zj: torch.Tensor, rij: torch.Tensor
    ) -> torch.Tensor:  # [eV / A]
        return -_e * m / (4 * pi * _eps0) * torch.div(torch.mul(zi, zj), rij**2)

    def dd_eij(
        self, zi: torch.Tensor, zj: torch.Tensor, rij: torch.Tensor
    ) -> torch.Tensor:  # [eV / A^2]
        return _e * m / (2 * pi * _eps0) * torch.div(torch.mul(zi, zj), rij**3)

    def switch_fn(
        self,
        zi: torch.Tensor,
        zj: torch.Tensor,
        rij: torch.Tensor,
        aij: torch.Tensor,
        router: torch.Tensor,
        rinner: torch.Tensor,
    ) -> torch.Tensor:  # [eV]
        # aij = self.a0 / (torch.pow(zi, self.p) + torch.pow(zj, self.p))

        xrouter = router / aij

        energy = self.eij(zi, zj, router) * self.phi(xrouter)

        grad1 = self.d_eij(zi, zj, router) * self.phi(xrouter) + self.eij(
            zi, zj, router
        ) * self.d_phi(xrouter)

        grad2 = (
            self.dd_eij(zi, zj, router) * self.phi(xrouter)
            + self.d_eij(zi, zj, router) * self.d_phi(xrouter)
            + self.d_eij(zi, zj, router) * self.d_phi(xrouter)
            + self.eij(zi, zj, router) * self.dd_phi(xrouter)
        )

        A = (-3 * grad1 + (router - rinner) * grad2) / (router - rinner) ** 2
        B = (2 * grad1 - (router - rinner) * grad2) / (router - rinner) ** 3
        C = (
            -energy
            + 1.0 / 2.0 * (router - rinner) * grad1
            - 1.0 / 12.0 * (router - rinner) ** 2 * grad2
        )

        switching = torch.where(
            rij < rinner,
            C,
            A / 3.0 * (rij - rinner) ** 3 + B / 4.0 * (rij - rinner) ** 4 + C,
        )

        return switching

    def envelope(self, r: torch.Tensor, rc: torch.Tensor, p: int = 6):
        x = r / rc
        y = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p)
            + p * (p + 2.0) * torch.pow(x, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(x, p + 2)
        ) * (x < 1)
        return y

    def _get_derivatives(self, energy: torch.Tensor, data: Data):
        egradi, egradij = torch.autograd.grad(
            outputs=[energy],  # TODO: generalized derivatives
            inputs=[data.positions, data.vij],  # TODO: generalized derivatives
            grad_outputs=[torch.ones_like(energy)],
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )

        volume = torch.det(data.cell)  # (batch,)
        rfaxy = torch.einsum("ax,ay->axy", data.vij, -egradij)

        edge_batch = data.batch[data.edge_index[0]]

        stress = (
            -0.5
            * torch_scatter.scatter_sum(rfaxy, edge_batch, dim=0)
            / volume.view(-1, 1)
        )

        return -egradi, stress

    def forward(
        self,
        data: Data,
    ) -> dict[str, torch.Tensor]:
        # TODO: generalized derivatives
        data.positions.requires_grad_(True)

        numbers = data.numbers  # (sum(N), )
        positions = data.positions  # (sum(N), 3)
        edge_index = data.edge_index  # (2, sum(E))
        edge_shift = data.edge_shift  # (sum(E), 3)
        batch = data.batch  # (sum(N), )

        edge_src, edge_dst = edge_index[0], edge_index[1]

        if "rij" not in data or "vij" not in data:
            data.vij = positions[edge_dst] - positions[edge_src] + edge_shift
            data.rij = LA.norm(data.vij, dim=-1)

        rbond = (
            self.covalent_radii[numbers[edge_src]]
            + self.covalent_radii[numbers[edge_dst]]
        )

        rij = data.rij
        zi = numbers[edge_src]  # (sum(E), )
        zj = numbers[edge_dst]  # (sum(E), )

        aij = self.a0 / (torch.pow(zi, self.p) + torch.pow(zj, self.p))  # (sum(E), )

        energy_pairs = (
            self.eij(zi, zj, rij)
            * self.phi(rij / aij.to(rij))
            * self.envelope(rij, torch.min(data.cutoff, rbond))
        )

        energy_nodes = 0.5 * torch_scatter.scatter_add(
            src=energy_pairs,
            index=edge_dst,
            dim=0,
        )  # (sum(N), )

        energies = torch_scatter.scatter_add(
            src=energy_nodes,
            index=batch,
            dim=0,
        )  # (B, )

        # TODO: generalized derivatives
        forces, stress = self._get_derivatives(energies, data)

        return {
            "energy": energies,
            "forces": forces,
            "stress": stress,
        }
