# Room-split push-optimal Sokoban solver

[![sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/0xyfann/makefun-sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/0xyfann/makefun-sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for make.fun's
["Sokoban solver"](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty. It reads each level from `/input/<id>/`, solves it with the
fewest pushes possible, and writes the move sequence to
`/output/<id>/`.

## How it works

The generator lays one hallway across the board with one to four rooms
off it, each room holding 3–4 boxes and as many goals — and a box pushed
out of its room can never come back, so the push-optimal count is the sum
of the rooms' optima. The solver therefore never searches the whole level:

1. Cut the level into hallway + rooms (one door each).
2. Solve every room on its own with an A* over pushes (not moves),
   using a Hungarian assignment heuristic, dead-square pruning from a
   reverse pull search, and 2×2 freeze pruning.
3. Order the rooms so the total push count is minimal.
4. Stitch the rooms together with BFS walks between doors and write the
   moves as LURD.

The work budget is counted in expansions, never in wall-clock time, so
two runs write byte-identical output. A greedy best-first fallback
covers the case where A* hits its expansion cap.

## Local results

- Practice set: **1,000,000 / 1,000,000** (all 5 instances push-optimal).
- 9 extra sets drawn with the kit's generator: **1,000,000 / 1,000,000**
  each (45/45 push-optimal).
- The house baseline (plain A*) scores 200,000 on the practice set.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see [LICENSE](LICENSE).
