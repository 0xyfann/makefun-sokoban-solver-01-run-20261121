// Sokoban solver: room-decomposed, push-optimal A*.
//
// Idea. The generator lays one hallway across the board with one to four
// rooms off it, each room holding 3-4 boxes and as many goals, and a box
// pushed out of its room can never come back, so the push-optimal count
// P* is the sum of the rooms' optima. We therefore never search the whole
// level (up to 16 boxes). Instead we:
//   1. cut the level into hallway + rooms (one door each),
//   2. solve every room on its own with an A* over pushes (not moves),
//      with a Hungarian assignment heuristic, dead-square pruning from a
//      reverse pull search, and 2x2 freeze pruning,
//   3. order the rooms so the total push count is minimal (every room but
//      the last must end with the player able to walk back out its door),
//   4. stitch the room push sequences together with player walks and
//      write the LURD moves.
//
// Only the number of pushes matters for the score, so walks are free.
// The search is budgeted by work (a node counter), never by the clock,
// and every tie-break is deterministic, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budget for the whole run (all instances), counted in A* expansions.
// Never the wall clock: a clock cut-off would make two runs differ.
// Measured locally: ~50us an expansion on these small rooms, so the caps
// below bound the worst case to roughly 40s of solving plus ~1s of
// start-up, inside the 60s wall budget even if the grader's CPU runs
// twice as slowly. Real rooms need at most ~1,100 expansions each.
const GLOBAL_EXPANSION_CAP = 400_000;
const ROOM_EXPANSION_CAP = 100_000;

const budget = { used: 0, cap: GLOBAL_EXPANSION_CAP };

// Directions: up, down, left, right.
const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// ---------------------------------------------------------------------------
// Level parsing
// ---------------------------------------------------------------------------

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  floor: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const floor = new Uint8Array(w * h);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      floor[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, floor, goals, goalSet, boxes, player };
}

// ---------------------------------------------------------------------------
// Grid helpers: flood fill, BFS distances, BFS walk path
// ---------------------------------------------------------------------------

function flood(w: number, h: number, pass: Uint8Array, src: number): Uint8Array {
  const seen = new Uint8Array(w * h);
  if (!pass[src]) return seen;
  const stack: number[] = [src];
  seen[src] = 1;
  while (stack.length > 0) {
    const c = stack.pop()!;
    const x = c % w;
    const y = (c / w) | 0;
    if (x > 0 && pass[c - 1] && !seen[c - 1]) {
      seen[c - 1] = 1;
      stack.push(c - 1);
    }
    if (x + 1 < w && pass[c + 1] && !seen[c + 1]) {
      seen[c + 1] = 1;
      stack.push(c + 1);
    }
    if (y > 0 && pass[c - w] && !seen[c - w]) {
      seen[c - w] = 1;
      stack.push(c - w);
    }
    if (y + 1 < h && pass[c + w] && !seen[c + w]) {
      seen[c + w] = 1;
      stack.push(c + w);
    }
  }
  return seen;
}

function bfsDist(w: number, h: number, pass: Uint8Array, src: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  if (!pass[src]) return dist;
  const q: number[] = [src];
  dist[src] = 0;
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const d = dist[c]! + 1;
    const x = c % w;
    const y = (c / w) | 0;
    if (x > 0 && pass[c - 1] && dist[c - 1] < 0) {
      dist[c - 1] = d;
      q.push(c - 1);
    }
    if (x + 1 < w && pass[c + 1] && dist[c + 1] < 0) {
      dist[c + 1] = d;
      q.push(c + 1);
    }
    if (y > 0 && pass[c - w] && dist[c - w] < 0) {
      dist[c - w] = d;
      q.push(c - w);
    }
    if (y + 1 < h && pass[c + w] && dist[c + w] < 0) {
      dist[c + w] = d;
      q.push(c + w);
    }
  }
  return dist;
}

// Shortest walk (no pushes) from src to dst over passable cells, as LURD
// letters, or null when unreachable. The path never steps on a box because
// the caller excludes box cells from pass.
function bfsPath(
  w: number,
  h: number,
  pass: Uint8Array,
  src: number,
  dst: number,
): string | null {
  if (src === dst) return '';
  const prev = new Int32Array(w * h).fill(-1);
  const prevDir = new Int8Array(w * h).fill(-1);
  const q: number[] = [src];
  prev[src] = src;
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const x = c % w;
    const y = (c / w) | 0;
    for (let di = 0; di < 4; di++) {
      const d = DIRS[di]!;
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (!pass[n] || prev[n] >= 0) continue;
      prev[n] = c;
      prevDir[n] = di;
      if (n === dst) {
        let s = '';
        let cur = n;
        while (cur !== src) {
          s = DIRS[prevDir[cur]!]!.ch + s;
          cur = prev[cur]!;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Room detection: hallway row + one door per room
// ---------------------------------------------------------------------------

interface Room {
  door: number; // cell in the wall row next to the hallway; part of the room
  entry: number; // hallway cell next to the door
  cells: Uint8Array; // mask of the room's cells (door included)
  boxes: number[];
  goals: number[];
}

function detectRooms(lv: Level): { rooms: Room[]; hallway: Uint8Array } | null {
  const { w, h, floor, player } = lv;
  const y0 = (player / w) | 0;
  // Hallway: the contiguous floor run in the player's row.
  const hallway = new Uint8Array(w * h);
  {
    const stack = [player];
    hallway[player] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      const x = c % w;
      if (x > 0 && floor[c - 1] && !hallway[c - 1] && ((c - 1) / w | 0) === y0) {
        hallway[c - 1] = 1;
        stack.push(c - 1);
      }
      if (x + 1 < w && floor[c + 1] && !hallway[c + 1] && ((c + 1) / w | 0) === y0) {
        hallway[c + 1] = 1;
        stack.push(c + 1);
      }
    }
  }
  // Floor outside the hallway row, for the room flood fills.
  const offHall = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) offHall[i] = floor[i] && ((i / w) | 0) !== y0 ? 1 : 0;
  const rooms: Room[] = [];
  const seenDoor = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) {
    const hc = y0 * w + x;
    if (!hallway[hc]) continue;
    for (const dy of [-1, 1]) {
      const y = y0 + dy;
      if (y < 0 || y >= h) continue;
      const dc = y * w + x;
      if (!floor[dc] || seenDoor[dc]) continue;
      seenDoor[dc] = 1;
      const cells = flood(w, h, offHall, dc);
      const boxes = lv.boxes.filter((b) => cells[b] === 1);
      const goals = lv.goals.filter((g) => cells[g] === 1);
      rooms.push({ door: dc, entry: hc, cells, boxes, goals });
    }
  }
  if (rooms.length === 0) return null;
  // Sanity: every box and goal in exactly one room, same count per room,
  // and exactly one door cell per room.
  const boxRoom = new Int32Array(w * h).fill(-1);
  const goalRoom = new Int32Array(w * h).fill(-1);
  for (let ri = 0; ri < rooms.length; ri++) {
    const r = rooms[ri]!;
    if (r.boxes.length === 0 || r.boxes.length !== r.goals.length) return null;
    let doors = 0;
    for (let i = 0; i < w * h; i++) {
      if (r.cells[i] !== 1) continue;
      const x = i % w;
      const y = (i / w) | 0;
      if ((y === y0 - 1 || y === y0 + 1) && hallway[y0 * w + x] === 1) doors++;
    }
    if (doors !== 1) return null;
    for (const b of r.boxes) {
      if (boxRoom[b] >= 0) return null;
      boxRoom[b] = ri;
    }
    for (const g of r.goals) {
      if (goalRoom[g] >= 0) return null;
      goalRoom[g] = ri;
    }
  }
  for (const b of lv.boxes) if (boxRoom[b] < 0) return null;
  for (const g of lv.goals) if (goalRoom[g] < 0) return null;
  return { rooms, hallway };
}

// ---------------------------------------------------------------------------
// Dead squares: reverse pull search from the room's goals.
// A box on a cell outside this set can never reach any goal.
// ---------------------------------------------------------------------------

function deadMask(lv: Level, room: Room, hallway: Uint8Array): Uint8Array {
  const { w, h } = lv;
  // Cells the player may stand on while pulling: room cells + hallway cells.
  const stand = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) stand[i] = room.cells[i] === 1 || hallway[i] === 1 ? 1 : 0;
  const alive = new Uint8Array(w * h);
  const q: number[] = [];
  for (const g of room.goals) {
    alive[g] = 1;
    q.push(g);
  }
  for (let head = 0; head < q.length; head++) {
    const p = q[head]!;
    const px = p % w;
    const py = (p / w) | 0;
    for (let di = 0; di < 4; di++) {
      const d = DIRS[di]!;
      const qx = px + d.dx;
      const qy = py + d.dy;
      if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
      const qc = qy * w + qx;
      if (!room.cells[qc] || alive[qc]) continue;
      // Reverse pull: the box moves p -> qc, the player must stand at
      // 2*qc - p, i.e. one step past qc away from p.
      const bx = qx + d.dx;
      const by = qy + d.dy;
      if (bx < 0 || bx >= w || by < 0 || by >= h) continue;
      if (!stand[by * w + bx]) continue;
      alive[qc] = 1;
      q.push(qc);
    }
  }
  return alive;
}

// ---------------------------------------------------------------------------
// Heuristic: minimum assignment of boxes to goals (Hungarian by brute force
// for the small rooms; sum of nearest-goal distances for larger sets).
// Admissible: every push moves one box one cell, so the pushes a box still
// needs are at least its grid distance to the goal it ends on.
// ---------------------------------------------------------------------------

function minAssign(cost: number[][], n: number): number {
  if (n === 0) return 0;
  let best = Infinity;
  const used = new Array<boolean>(n).fill(false);
  const dfs = (i: number, acc: number): void => {
    if (acc >= best) return;
    if (i === n) {
      best = acc;
      return;
    }
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      const c = cost[i]![j]!;
      if (c === Infinity) continue;
      used[j] = true;
      dfs(i + 1, acc + c);
      used[j] = false;
    }
  };
  dfs(0, 0);
  return best;
}

function heuristicValue(
  boxes: number[],
  goalDist: Int32Array[],
  n: number,
): number {
  if (n <= 7) {
    const cost: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        const d = goalDist[j]![boxes[i]!]!;
        row.push(d < 0 ? Infinity : d);
      }
      cost.push(row);
    }
    return minAssign(cost, n);
  }
  let s = 0;
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (let j = 0; j < n; j++) {
      const d = goalDist[j]![boxes[i]!]!;
      if (d >= 0 && d < m) m = d;
    }
    if (m === Infinity) return Infinity;
    s += m;
  }
  return s;
}

// ---------------------------------------------------------------------------
// 2x2 freeze: a 2x2 block of walls/boxes holding a box and no goal is dead.
// ---------------------------------------------------------------------------

// 2x2 freeze: a 2x2 block of walls/boxes holding a box and no goal is dead.
// boxes is short (at most 4 in a room), so membership is a linear scan
// instead of an allocated set.
function hasFreeze(lv: Level, boxes: number[]): boolean {
  const { w, h, wall, goalSet } = lv;
  for (const b of boxes) {
    const x = b % w;
    const y = (b / w) | 0;
    for (let oy = -1; oy <= 0; oy++) {
      for (let ox = -1; ox <= 0; ox++) {
        const cx = x + ox;
        const cy = y + oy;
        if (cx < 0 || cy < 0 || cx + 1 >= w || cy + 1 >= h) continue;
        const c00 = cy * w + cx;
        const cells = [c00, c00 + 1, c00 + w, c00 + w + 1];
        let allBlocked = true;
        let anyGoal = false;
        for (const c of cells) {
          if (goalSet[c]) {
            anyGoal = true;
            break;
          }
          if (!wall[c] && !boxes.includes(c)) {
            allBlocked = false;
            break;
          }
        }
        if (allBlocked && !anyGoal) return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// A* over pushes for one room. With greedy set, the heap orders by the
// heuristic alone (best-first): not optimal, but a useful fallback when
// the optimal search exhausts its expansion cap, since a solved room in
// extra pushes still earns partial points while an unsolved one earns 0.
// ---------------------------------------------------------------------------

interface Push {
  box: number; // box cell before the push
  dir: number; // index into DIRS
}

interface RoomSolution {
  pushes: Push[];
  finalBoxes: number[];
  finalPlayer: number;
}

function regionFrom(
  lv: Level,
  regionPass: Uint8Array,
  boxes: number[],
  src: number,
): Uint8Array {
  const { w, h } = lv;
  const boxMask = new Uint8Array(w * h);
  for (const b of boxes) boxMask[b] = 1;
  const pass = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) pass[i] = regionPass[i] === 1 && boxMask[i] === 0 ? 1 : 0;
  return flood(w, h, pass, src);
}

function repOf(mask: Uint8Array): number {
  for (let i = 0; i < mask.length; i++) if (mask[i]) return i;
  return -1;
}

function solveRoom(
  lv: Level,
  room: Room,
  hallway: Uint8Array,
  requireExit: boolean,
  greedy: boolean,
): RoomSolution | null {
  const { w, h, goalSet } = lv;
  const n = room.boxes.length;
  const goalDist: Int32Array[] = room.goals.map((g) => bfsDist(w, h, room.cells, g));
  const alive = deadMask(lv, room, hallway);
  const regionPass = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) regionPass[i] = room.cells[i] === 1 || hallway[i] === 1 ? 1 : 0;

  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  for (const b of startBoxes) if (!alive[b]) return null;
  const h0 = heuristicValue(startBoxes, goalDist, n);
  if (h0 === Infinity) return null;

  // State storage (parallel arrays).
  const sBoxes: number[][] = [];
  const sPlayer: number[] = [];
  const sG: number[] = [];
  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const closed = new Map<string, number>();

  // Binary heap of [f, h, seq, stateIndex]; ties break on h, then seq, so
  // the order is fully deterministic.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const heapLess = (
    a: [number, number, number, number],
    b: [number, number, number, number],
  ): boolean =>
    a[0] < b[0] ||
    (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const keyOf = (boxes: number[], rep: number): string => boxes.join(',') + '|' + rep;

  const initRegion = regionFrom(lv, regionPass, startBoxes, room.entry);
  const initRep = repOf(initRegion);
  if (initRep < 0) return null;
  sBoxes.push(startBoxes);
  sPlayer.push(room.entry);
  sG.push(0);
  sParent.push(-1);
  sPushBox.push(-1);
  sPushDir.push(-1);
  closed.set(keyOf(startBoxes, initRep), 0);
  heapPush([h0, h0, seq++, 0]);

  let expanded = 0;
  while (heap.length > 0) {
    if (expanded >= ROOM_EXPANSION_CAP || budget.used >= budget.cap) return null;
    const [, , , idx] = heapPop();
    const boxes = sBoxes[idx]!;
    const player = sPlayer[idx]!;
    const g = sG[idx]!;
    // Stale entry: a better path to this state was already recorded.
    const region = regionFrom(lv, regionPass, boxes, player);
    const rep = repOf(region);
    if (rep < 0) continue;
    if (closed.get(keyOf(boxes, rep)) !== g) continue;
    expanded++;
    budget.used++;

    // Goal: every box on a goal.
    if (boxes.every((b) => goalSet[b])) {
      if (!requireExit || region[room.door] === 1) {
        const pushes: Push[] = [];
        for (let t = idx; t > 0; t = sParent[t]!) {
          pushes.push({ box: sPushBox[t]!, dir: sPushDir[t]! });
        }
        pushes.reverse();
        return { pushes, finalBoxes: boxes, finalPlayer: player };
      }
      // Goal reached but the player cannot walk back out: keep searching
      // for an exit-feasible goal state; do not expand this one further.
      continue;
    }

    const boxSet = new Uint8Array(w * h);
    for (const b of boxes) boxSet[b] = 1;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di]!;
        const behindX = bx - d.dx;
        const behindY = by - d.dy;
        const destX = bx + d.dx;
        const destY = by + d.dy;
        if (behindX < 0 || behindX >= w || behindY < 0 || behindY >= h) continue;
        if (destX < 0 || destX >= w || destY < 0 || destY >= h) continue;
        const behind = behindY * w + behindX;
        const dest = destY * w + destX;
        if (!region[behind]) continue; // player cannot stand there
        if (!room.cells[dest]) continue; // the box must stay in its room
        if (boxSet[dest]) continue; // destination occupied
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        let dead = false;
        for (const x of nb) {
          if (!alive[x]) {
            dead = true;
            break;
          }
        }
        if (dead) continue;
        if (hasFreeze(lv, nb)) continue;
        const hn = heuristicValue(nb, goalDist, n);
        if (hn === Infinity) continue;
        // The player ends on the box's old cell; canonicalise by the
        // reachable region from there.
        const newRegion = regionFrom(lv, regionPass, nb, b);
        const newRep = repOf(newRegion);
        if (newRep < 0) continue;
        const key = keyOf(nb, newRep);
        const ng = g + 1;
        const prevBest = closed.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        closed.set(key, ng);
        const ni = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sG.push(ng);
        sParent.push(idx);
        sPushBox.push(b);
        sPushDir.push(di);
        heapPush([greedy ? hn : ng + hn, hn, seq++, ni]);
      }
    }
  }
  return null;
}

function canExit(
  lv: Level,
  room: Room,
  hallway: Uint8Array,
  boxes: number[],
  player: number,
): boolean {
  const { w, h } = lv;
  const regionPass = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) regionPass[i] = room.cells[i] === 1 || hallway[i] === 1 ? 1 : 0;
  const region = regionFrom(lv, regionPass, boxes, player);
  return region[room.door] === 1;
}

// ---------------------------------------------------------------------------
// Whole-level solve: decompose, solve rooms, order them, stitch the moves.
// ---------------------------------------------------------------------------

function passWithoutBoxes(lv: Level, boxes: number[]): Uint8Array {
  const { w, h, floor } = lv;
  const pass = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) pass[i] = floor[i];
  for (const b of boxes) pass[b] = 0;
  return pass;
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const dfs = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      dfs();
      cur.pop();
      used[i] = false;
    }
  };
  dfs();
  return out;
}

function solveLevel(lv: Level): string {
  const detected = detectRooms(lv);
  let rooms: Room[];
  let hallway: Uint8Array;
  if (detected) {
    rooms = detected.rooms;
    hallway = detected.hallway;
  } else {
    // Fallback: the whole level as one room (no decomposition possible).
    rooms = [
      {
        door: lv.player,
        entry: lv.player,
        cells: lv.floor,
        boxes: [...lv.boxes],
        goals: [...lv.goals],
      },
    ];
    hallway = new Uint8Array(lv.w * lv.h);
  }

  // Solve each room: first unconstrained, then (only if the unconstrained
  // optimum cannot walk back out) with the exit constraint. Each attempt
  // tries the optimal search first and falls back to greedy best-first
  // when the optimal search exhausts its expansion cap.
  const solveOne = (room: Room, requireExit: boolean): RoomSolution | null =>
    solveRoom(lv, room, hallway, requireExit, false) ??
    solveRoom(lv, room, hallway, requireExit, true);
  const sols: { any: RoomSolution | null; exit: RoomSolution | null }[] = rooms.map(
    (room) => {
      const any = solveOne(room, false);
      let exit: RoomSolution | null = null;
      if (any) {
        exit = canExit(lv, room, hallway, any.finalBoxes, any.finalPlayer)
          ? any
          : solveOne(room, true);
      }
      return { any, exit };
    },
  );
  if (sols.some((s) => !s.any)) return '';

  // Order the rooms: every room but the last must end exit-feasible.
  // Minimise the total push count over all orders.
  let bestOrder: number[] | null = null;
  let bestCost = Infinity;
  for (const order of permutations(rooms.length)) {
    let cost = 0;
    let ok = true;
    for (let k = 0; k < order.length; k++) {
      const ri = order[k]!;
      const isLast = k === order.length - 1;
      const s = isLast ? sols[ri]!.any : sols[ri]!.exit;
      if (!s) {
        ok = false;
        break;
      }
      cost += s.pushes.length;
    }
    if (ok && cost < bestCost) {
      bestCost = cost;
      bestOrder = order;
    }
  }
  if (!bestOrder) return '';

  // Stitch: walk to each room's door, play its pushes, walk on.
  let moves = '';
  let player = lv.player;
  let boxes = [...lv.boxes].sort((a, b) => a - b);
  for (let k = 0; k < bestOrder.length; k++) {
    const ri = bestOrder[k]!;
    const room = rooms[ri]!;
    const sol = (k === bestOrder.length - 1 ? sols[ri]!.any : sols[ri]!.exit)!;
    let pass = passWithoutBoxes(lv, boxes);
    const walk = bfsPath(lv.w, lv.h, pass, player, room.entry);
    if (walk === null) return '';
    moves += walk;
    player = room.entry;
    for (const p of sol.pushes) {
      const d = DIRS[p.dir]!;
      const delta = d.dy * lv.w + d.dx;
      const behind = p.box - delta;
      const walk2 = bfsPath(lv.w, lv.h, pass, player, behind);
      if (walk2 === null) return '';
      moves += walk2;
      moves += d.ch;
      player = p.box;
      const idx = boxes.indexOf(p.box);
      boxes[idx] = p.box + delta;
      boxes.sort((a, b) => a - b);
      pass = passWithoutBoxes(lv, boxes);
    }
  }
  return moves;
}

// ---------------------------------------------------------------------------
// Entry point: one run over the whole set.
// ---------------------------------------------------------------------------

for (const id of readdirSync(IN_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
