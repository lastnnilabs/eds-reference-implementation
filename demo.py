import numpy as np
from typing import Dict, Optional

np.random.seed(42)

THETA = 0.3
VEC_DIM = 8


class Dimension:
    def __init__(self, name: str, level: int):
        self.name = name
        self.level = level
        self.vec = np.random.randn(VEC_DIM)


class Point:
    def __init__(self, name: str, dims: Dict[str, float]):
        self.name = name
        self.dims = dims

    def __repr__(self):
        return f"Point({self.name}, dim={len(self.dims)})"


GDS: Dict[str, Dimension] = {
    name: Dimension(name, level) for name, level in [
        ("fur", 0), ("warm", 0), ("sound", 0), ("soft", 0),
        ("fuzzy", 0), ("quiet", 0), ("loud", 0),
        ("living", 1), ("category", 1), ("mammal", 1),
        ("entity", 1), ("feeling", 1),
    ]
}

cat     = Point("cat",     {"fur": 0.9, "warm": 0.8, "living": 0.7, "category": 0.6})
animal  = Point("animal",  {"living": 0.9, "category": 0.8})
warmth  = Point("warmth",  {"warm": 0.95})
purring = Point("purring", {"sound": 0.85, "warm": 0.5})


def centroid(p: Point) -> np.ndarray:
    vecs = np.array([GDS[d].vec for d in p.dims])
    weights = np.array(list(p.dims.values()))
    return (vecs * weights[:, None]).sum(axis=0) / weights.sum()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sim(p: Point, q: Point) -> float:
    shared = set(p.dims) & set(q.dims)
    if not shared:
        return 0.0
    num = sum(min(p.dims[d], q.dims[d]) for d in shared)
    den = sum(max(p.dims.get(d, 0), q.dims.get(d, 0)) for d in set(p.dims) | set(q.dims))
    return num / den


def compatible(dim_name: str, p: Point, c: Optional[np.ndarray] = None) -> bool:
    d = GDS[dim_name]
    max_level = max(GDS[k].level for k in p.dims) if p.dims else 0
    if d.level > max_level:
        return False
    return cos_sim(d.vec, c if c is not None else centroid(p)) >= THETA


def transition(p: Point, q: Point) -> bool:
    deficit = set(q.dims) - set(p.dims)
    if not deficit:
        return True
    c = centroid(p)
    return all(compatible(d, p, c) for d in deficit)


def analyze_transition(p: Point, q: Point) -> None:
    deficit = set(q.dims) - set(p.dims)
    max_level = max(GDS[k].level for k in p.dims) if p.dims else 0
    c = centroid(p)

    print(f"\n=== Analysis: T({p.name} -> {q.name}) ===")
    print(f"Deficit: {deficit}")
    print(f"max_level({p.name}) = {max_level}")
    print(f"Threshold θ = {THETA}\n")

    for d in sorted(deficit):
        dim = GDS[d]
        cs = cos_sim(dim.vec, c)
        cond_c = dim.level <= max_level
        cond_b = cs >= THETA
        status = "OK" if (cond_c and cond_b) else "blocked"
        print(f"  {d:10s}  level={dim.level}  cos_sim={cs:+.4f}  "
              f"C:{'PASS' if cond_c else 'FAIL'}  B:{'PASS' if cond_b else 'FAIL'}  → {status}")

    result = transition(p, q)
    print(f"\nResult: T({p.name} -> {q.name}) = {'OK' if result else 'FAIL'}")


if __name__ == "__main__":
    pairs = [(cat, animal), (cat, warmth), (cat, purring), (warmth, purring)]
    transitions = [(warmth, cat), (warmth, purring), (purring, cat), (animal, cat)]

    print("=== Structural Similarity ===")
    for a, b in pairs:
        print(f"sim({a.name}, {b.name}){' ' * (10 - len(a.name) - len(b.name))}= {sim(a, b):.2f}")

    print("\n=== Transitions ===")
    for a, b in transitions:
        print(f"T({a.name} -> {b.name}){' ' * (10 - len(a.name) - len(b.name))}= {'OK' if transition(a, b) else 'FAIL'}")

    print("\n=== Dimensionality ===")
    for p in [cat, animal, warmth, purring]:
        print(f"dim({p.name}){' ' * (8 - len(p.name))}= {len(p.dims)}")

    analyze_transition(purring, cat)
