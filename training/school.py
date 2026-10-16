"""Vectorized NumPy simulation of one representative fish group (training only).

It mirrors the browser simulation (boids + predators + food + group strategy) at a smaller
scale. Neighbour search is brute force, which is fine for ~50 fish.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.config import EnvConfig

EPS = 1e-6
PURSUIT_RADIUS = 12.0  # mirrors frontend/src/sim/predator.ts


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > max_norm, v * (max_norm / np.maximum(n, EPS)), v)


def steer_towards(direction: np.ndarray, vel: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Reynolds steering; zero directions give zero steering."""
    n = np.linalg.norm(direction, axis=-1, keepdims=True)
    desired = np.where(n > EPS, direction / np.maximum(n, EPS) * cfg.max_speed, vel)
    return clamp_norm(desired - vel, cfg.max_force)


@dataclass
class StepEvents:
    food: float = 0.0
    caught: int = 0
    collisions: float = 0


class SchoolSim:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator) -> None:
        self.cfg, self.rng = cfg, rng
        self.half = np.array([50.0, 25.0, 50.0])
        n = cfg.n_fish
        center = rng.uniform(-0.5, 0.5, 3) * self.half
        self.pos = center + rng.uniform(-6, 6, (n, 3))
        self.vel = rng.normal(size=(n, 3)) * cfg.min_speed
        self.energy = np.full(n, 0.5)
        self.alive = np.ones(n, dtype=bool)
        self.pred_pos = rng.uniform(-0.8, 0.8, (rng.integers(cfg.predators[0], cfg.predators[1] + 1), 3)) * self.half
        self.pred_vel = np.zeros_like(self.pred_pos)
        self.pred_speed = rng.uniform(*cfg.predator_speed)
        self.digest = np.zeros(len(self.pred_pos))
        self.food_pos = np.empty((0, 3))
        self.food_amount = np.empty(0)
        self._spawn_food(rng.integers(cfg.food[0], cfg.food[1] + 1))
        # strategy: direction (3) + weights (pred, food, coh)
        self.direction = np.zeros(3)
        self.weights = np.array([0.6, 0.3, 0.5])

    # ---- public -------------------------------------------------------------------------
    def set_strategy(self, direction: np.ndarray, weights: np.ndarray) -> None:
        self.direction, self.weights = direction.astype(float), weights.astype(float)

    def step(self, dt: float) -> StepEvents:
        events = StepEvents()
        acc, events.collisions = self._fish_acceleration()
        self._integrate_fish(acc, dt)
        self._move_predators(dt)
        events.caught = self._predation()
        events.food = self._feed(dt)
        return events

    def center(self) -> np.ndarray:
        return self.pos[self.alive].mean(axis=0) if self.alive.any() else np.zeros(3)

    def dispersion(self) -> float:
        if not self.alive.any():
            return 0.0
        return float(np.linalg.norm(self.pos[self.alive] - self.center(), axis=1).mean())

    # ---- fish ---------------------------------------------------------------------------
    def _fish_acceleration(self) -> tuple[np.ndarray, int]:
        c = self.cfg
        idx = np.flatnonzero(self.alive)
        p, v = self.pos[idx], self.vel[idx]
        d = p[None, :, :] - p[:, None, :]  # d[i, j] = p_j - p_i
        dist = np.linalg.norm(d, axis=-1) + np.eye(len(idx)) * 1e9
        near = dist < c.perception
        sep_mask = dist < c.separation_radius
        collisions = int(np.triu(dist < c.collision_radius).sum())
        acc = c.w_sep * self._separation(d, dist, sep_mask, v)
        acc += c.w_align * self._alignment(near, v)
        acc += c.w_coh * 2 * self.weights[2] * self._cohesion(near, p, v)
        acc += c.w_boundary * self._boundary(p, v)
        acc += c.w_predator * self.weights[0] * self._flee(p)
        acc += c.w_food * self.weights[1] * self._seek_food(p, idx)
        acc += c.w_ml * self._group_direction(v, p)
        out = np.zeros_like(self.pos)
        out[idx] = clamp_norm(acc, c.max_force)
        return out, collisions

    def _separation(self, d, dist, mask, v):
        push = -(d / np.maximum(dist, EPS)[..., None] ** 2) * mask[..., None]
        total = push.sum(axis=1)
        return np.where(mask.any(axis=1, keepdims=True), steer_towards(total, v, self.cfg), 0.0)

    def _alignment(self, near, v):
        mean_v = (near[..., None] * v[None]).sum(axis=1)
        return np.where(near.any(axis=1, keepdims=True), steer_towards(mean_v, v, self.cfg), 0.0)

    def _cohesion(self, near, p, v):
        count = near.sum(axis=1, keepdims=True)
        centroid = (near[..., None] * p[None]).sum(axis=1) / np.maximum(count, 1)
        return np.where(count > 0, steer_towards(centroid - p, v, self.cfg), 0.0)

    def _boundary(self, p, v):
        inner = self.half - self.cfg.boundary_margin
        push = np.where(p > inner, -np.minimum(1, (p - inner) / self.cfg.boundary_margin), 0.0)
        push += np.where(p < -inner, np.minimum(1, (-inner - p) / self.cfg.boundary_margin), 0.0)
        depth = np.abs(push).max(axis=1, keepdims=True)
        return steer_towards(push, v, self.cfg) * depth

    def _remove_outward(self, vec, p):
        inner = self.half - self.cfg.boundary_margin
        push = np.where(p > inner, -np.minimum(1, (p - inner) / self.cfg.boundary_margin), 0.0)
        push += np.where(p < -inner, np.minimum(1, (-inner - p) / self.cfg.boundary_margin), 0.0)
        outward = np.sign(vec) == -np.sign(push)
        return np.where(outward & (push != 0), vec * (1 - np.abs(push)), vec)

    def _flee(self, p):
        c = self.cfg
        rel = p[:, None, :] - self.pred_pos[None]
        dist = np.linalg.norm(rel, axis=-1)
        k = np.where(dist < c.danger_radius, c.max_force * (1 - dist / c.danger_radius) / np.maximum(dist, EPS), 0.0)
        return self._remove_outward((rel * k[..., None]).sum(axis=1), p)

    def _seek_food(self, p, idx):
        active = self.food_amount > 0
        if not active.any():
            return np.zeros_like(p)
        rel = self.food_pos[active][None] - p[:, None, :]
        dist = np.linalg.norm(rel, axis=-1)
        nearest = dist.argmin(axis=1)
        d = dist[np.arange(len(p)), nearest]
        r = rel[np.arange(len(p)), nearest]
        hunger = 1 - self.energy[idx]
        k = np.where(d < self.cfg.sense_radius, self.cfg.max_force * (0.25 + 0.75 * hunger) / np.maximum(d, EPS), 0.0)
        return r * k[:, None]

    def _group_direction(self, v, p):
        n = np.linalg.norm(self.direction)
        if n < 1e-3:
            return np.zeros_like(v)
        desired = self._remove_outward(np.broadcast_to(self.direction / n, v.shape).copy(), p)
        steer = desired * self.cfg.max_speed - v
        sn = np.linalg.norm(steer, axis=1, keepdims=True)
        return np.where(sn > EPS, steer / np.maximum(sn, EPS), 0.0) * self.cfg.max_force * min(1.0, n)

    def _integrate_fish(self, acc, dt):
        c = self.cfg
        v = self.vel + acc * dt
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(speed > EPS, v / np.maximum(speed, EPS), self.vel) * np.clip(speed, c.min_speed, c.max_speed)
        self.vel = np.where(self.alive[:, None], v, 0.0)
        self.pos = np.clip(self.pos + self.vel * dt, -self.half * 1.1, self.half * 1.1)
        self.energy = np.where(self.alive, np.maximum(0, self.energy - c.energy_decay * dt), self.energy)

    # ---- predators & food --------------------------------------------------------------
    def _move_predators(self, dt):
        to = self._predator_targets() - self.pred_pos
        dist = np.linalg.norm(to, axis=1, keepdims=True)
        desired = to / np.maximum(dist, EPS) * self.pred_speed
        self.digest = np.maximum(0, self.digest - dt)
        self.pred_vel += np.clip(desired - self.pred_vel, -12 * dt, 12 * dt)
        slow = np.where(self.digest > 0, 0.4, 1.0)[:, None]
        self.pred_pos = np.clip(self.pred_pos + self.pred_vel * dt * slow, -self.half, self.half)

    def _predator_targets(self) -> np.ndarray:
        """Group centre from afar; the nearest fish once within the pursuit radius."""
        center = self.center()
        targets = np.repeat(center[None], len(self.pred_pos), axis=0)
        if not self.alive.any():
            return targets
        fish = self.pos[self.alive]
        for k, p in enumerate(self.pred_pos):
            dist = np.linalg.norm(fish - p, axis=1)
            j = int(dist.argmin())
            if dist[j] < PURSUIT_RADIUS:
                targets[k] = fish[j]
        return targets

    def _predation(self) -> int:
        caught = 0
        for k in range(len(self.pred_pos)):
            if self.digest[k] > 0 or not self.alive.any():
                continue
            dist = np.where(self.alive, np.linalg.norm(self.pos - self.pred_pos[k], axis=1), np.inf)
            victim = int(dist.argmin())
            if dist[victim] < self.cfg.catch_radius:
                self.alive[victim] = False
                self.digest[k] = self.cfg.digest_time
                caught += 1
        return caught

    def _feed(self, dt) -> float:
        total = 0.0
        for f in np.flatnonzero(self.food_amount > 0):
            inside = self.alive & (np.linalg.norm(self.pos - self.food_pos[f], axis=1) < self.cfg.eat_radius)
            bites = np.minimum(self.cfg.eat_rate * dt, 1 - self.energy) * inside
            scale = min(1.0, self.food_amount[f] / max(bites.sum(), EPS))
            self.energy += bites * scale
            self.food_amount[f] -= bites.sum() * scale
            total += bites.sum() * scale
        if (self.food_amount <= 1e-9).any():
            self._respawn_empty()
        return float(total)

    def _spawn_food(self, n: int) -> None:
        self.food_pos = self.rng.uniform(-0.75, 0.75, (n, 3)) * self.half
        self.food_amount = np.full(n, self.cfg.food_amount)

    def _respawn_empty(self) -> None:
        empty = self.food_amount <= 1e-9
        self.food_pos[empty] = self.rng.uniform(-0.75, 0.75, (int(empty.sum()), 3)) * self.half
        self.food_amount[empty] = self.cfg.food_amount
