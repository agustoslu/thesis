"""adapted from https://github.com/google-research/DP-FTRL/tree/main"""

import torch
from collections import namedtuple
import flower
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Any, Dict


class CummulativeNoise:
    @torch.no_grad()
    def __init__(
        self,
        std: float,
        shapes: List[Tuple[int, ...]],
        device: torch.device,
        test_mode: bool = False,
    ) -> None:
        """
        std: standard deviation of each Gaussian noise
        shapes: a list of tensor shapes (matching gradient shapes)
        device: device to store the noise tensors
        test_mode: if True, we assume each draw of noise is all-ones
        (so we can count set-bits in binary)
        """
        assert std >= 0
        self.std: float = std
        self.shapes: List[Tuple[int, ...]] = shapes
        self.device: torch.device = device
        self.step: int = 0
        self.binary: List[int] = [0]
        self.noise_sum: List[torch.Tensor] = [
            torch.zeros(shape).to(self.device) for shape in shapes
        ]
        self.recorded: List[List[torch.Tensor]] = [
            [torch.zeros(shape).to(self.device) for shape in shapes]
        ]
        self.test_mode: bool = test_mode

    @torch.no_grad()
    def __call__(self) -> List[torch.Tensor]:
        """
        called whenever algorithm wants current noisy prefix-sum
        - update bits
        - if a bit flips from 1 to 0, we remove the noise from the sum
        - if a bit flips from 0 to 1, we add a new noise to the sum
        return running sum of all active internal-node noise
        like a sliding window of bits with merging in between
        """
        if self.std <= 0 and not self.test_mode:
            return self.noise_sum

        self.step += 1

        idx: int = 0
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns -= re
            idx += 1
        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append(
                [torch.zeros(shape).to(self.device) for shape in self.shapes]
            )

        for shape, ns, re in zip(self.shapes, self.noise_sum, self.recorded[idx]):
            if not self.test_mode:
                n: torch.Tensor = torch.normal(0, self.std, shape).to(
                    self.device
                )  # draw a new Gaussian
            else:
                n: torch.Tensor = torch.ones(shape).to(self.device)
            ns += n
            re.copy_(n)

        self.binary[idx] = 1
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target: int) -> List[torch.Tensor]:
        if self.step >= step_target:
            raise ValueError(f"Already reached {step_target}")
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


Element = namedtuple("Element", "height value")


class HonakerCummulativeNoise:
    """
    stack version, based on the tree aggreation trick of Honaker, "Efficient Use of Differentially Private Binary Trees", 2015
    """

    @torch.no_grad()
    def __init__(
        self,
        std: float,
        shapes: List[Tuple[int, ...]],
        device: torch.device,
        test_mode: bool = False,
    ) -> None:
        self.std: float = std
        self.shapes: List[Tuple[int, ...]] = shapes
        self.device: torch.device = device
        self.step: int = 0
        self.noise_sum: List[torch.Tensor] = [
            torch.zeros(shape).to(self.device) for shape in shapes
        ]
        self.stack: List[Any] = []

    @torch.no_grad()
    def get_noise(self) -> List[torch.Tensor]:
        return [
            torch.normal(0, self.std, shape).to(self.device) for shape in self.shapes
        ]

    @torch.no_grad()
    def push(self, elem: Any) -> None:
        for i in range(len(self.shapes)):
            self.noise_sum[i] += elem.value[i] / (2.0 - 1 / 2**elem.height)
        self.stack.append(elem)

    @torch.no_grad()
    def pop(self) -> None:
        elem = self.stack.pop()
        for i in range(len(self.shapes)):
            self.noise_sum[i] -= elem.value[i] / (2.0 - 1 / 2**elem.height)

    @torch.no_grad()
    def __call__(self) -> List[torch.Tensor]:
        self.step += 1
        self.push(Element(0, self.get_noise()))

        while len(self.stack) >= 2 and self.stack[-1].height == self.stack[-2].height:
            # create new element
            left_value, right_value = self.stack[-2].value, self.stack[-1].value
            new_noise = self.get_noise()
            new_elem = Element(
                self.stack[-1].height + 1,
                [
                    x + (y + z) / 2
                    for x, y, z in zip(new_noise, left_value, right_value)
                ],
            )

            self.pop()
            self.pop()

            self.push(new_elem)
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target: int) -> List[torch.Tensor]:
        if self.step >= step_target:
            raise ValueError(f"Already reached {step_target}")
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


class FTRLOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params: Any, momentum: float, record_last_noise: bool = True
    ) -> None:
        self.momentum: float = momentum
        self.record_last_noise: bool = record_last_noise
        super(FTRLOptimizer, self).__init__(params, dict())

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(FTRLOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(
        self, args: Tuple[float, List[torch.Tensor]], closure: Optional[Any] = None
    ) -> Optional[float]:
        alpha, noise = args
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, nz in zip(group["params"], noise):
                if p.grad is None:
                    continue

                d_p: torch.Tensor = p.grad

                param_state: Dict[str, Any] = self.state[p]

                if len(param_state) == 0:
                    param_state["grad_sum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    param_state["model_sum"] = p.detach().clone()(
                        memory_format=torch.preserve_format
                    )  # only record the initial model
                    param_state["momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if self.record_last_noise:
                        param_state["last_noise"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )  # record the last noise for restart

                gs: torch.Tensor = param_state["grad_sum"]
                ms: torch.Tensor = param_state["model_sum"]
                if self.momentum == 0:
                    gs.add(d_p)
                    p.copy_(ms + (-gs - nz) / alpha)
                else:
                    gs.add_(d_p)
                    param_state["momentum"].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms + param_state["momentum"] / alpha)
                if self.record_last_noise:
                    param_state["last_noise"].copy_(nz)
        return loss

    @torch.no_grad()
    def restart(self, last_noise: Optional[List[torch.Tensor]] = None) -> None:
        assert last_noise is not None or self.record_last_noise
        for group in self.param_groups:
            if last_noise is None:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    param_state: Dict[str, Any] = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state["grad_sum"].add_(param_state["last_noise"])
            else:
                for p, nz in zip(group["params"], last_noise):
                    if p.grad is None:
                        continue
                    param_state: Dict[str, Any] = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state["grad_sum"].add_(nz)
