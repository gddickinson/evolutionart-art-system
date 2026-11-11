// Python's random module and numpy's legacy np.random, ported so a seed gives the same art as the Python package.
// Both use MT19937; they differ in seeding (random.seed(int) -> init_by_array, np.random.seed(int) -> init_genrand).
// Also: Python float formatting (repr), float modulo and colorsys.hsv_to_rgb, which the engines need.

const N = 624;
const mul = (a, b) => Math.imul(a, b) >>> 0;

class MT19937 {
  constructor() { this.mt = new Uint32Array(N); this.mti = N + 1; }
  initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) mt[i] = (mul(1812433253, mt[i - 1] ^ (mt[i - 1] >>> 30)) + i) >>> 0;
    this.mti = N;
  }
  initByArray(key) {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1, j = 0;
    for (let k = Math.max(N, key.length); k; k--) {
      mt[i] = ((mt[i] ^ mul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k; k--) {
      mt[i] = ((mt[i] ^ mul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
  }
  uint32() {
    const mt = this.mt;
    if (this.mti >= N) {
      for (let kk = 0; kk < N; kk++) {
        const y = (mt[kk] & 0x80000000) | (mt[(kk + 1) % N] & 0x7fffffff);
        mt[kk] = mt[(kk + 397) % N] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  }
  // genrand_res53: a double in [0, 1) with 53 random bits (same in CPython and numpy).
  res53() { const a = this.uint32() >>> 5, b = this.uint32() >>> 6; return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0); }
}

// CPython random.Random: random(), uniform(), randint(), choice(), sample() with CPython 3.12's exact algorithms.
export class PyRandom {
  constructor(seed = 0) { this.gen = new MT19937(); this.seed(seed); }
  seed(n) {
    const key = []; let s = Math.abs(Math.trunc(n));
    do { key.push(s % 4294967296 >>> 0); s = Math.floor(s / 4294967296); } while (s > 0);
    this.gen.initByArray(key);
  }
  random() { return this.gen.res53(); }
  uniform(a, b) { return a + (b - a) * this.random(); }
  getrandbits(k) { return k <= 0 ? 0 : this.gen.uint32() >>> (32 - k); }   // k <= 32 is all the engines need
  randbelow(n) {
    const k = 32 - Math.clz32(n);                                              // n.bit_length()
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }
  randint(a, b) { return a + this.randbelow(b - a + 1); }
  choice(seq) {
    if (!seq.length) throw new Error('Cannot choose from an empty sequence');
    return seq[this.randbelow(seq.length)];
  }
  sample(population, k) {
    const n = population.length;
    if (k < 0 || k > n) throw new Error('Sample larger than population or is negative');
    let setsize = 21;
    if (k > 5) setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    const result = new Array(k);
    if (n <= setsize) {
      const pool = [...population];
      for (let i = 0; i < k; i++) {
        const j = this.randbelow(n - i);
        result[i] = pool[j];
        pool[j] = pool[n - i - 1];
      }
    } else {
      const selected = new Set();
      for (let i = 0; i < k; i++) {
        let j = this.randbelow(n);
        while (selected.has(j)) j = this.randbelow(n);
        selected.add(j);
        result[i] = population[j];
      }
    }
    return result;
  }
}

// numpy's legacy global RandomState (np.random.seed / uniform / normal).
export class NpRandom {
  constructor(seed = 0) { this.gen = new MT19937(); this.seed(seed); }
  seed(n) { this.gen.initGenrand(n >>> 0); this.hasGauss = false; this.gauss = 0; }
  randomSample() { return this.gen.res53(); }
  uniform(low, high, size) {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) out[i] = low + (high - low) * this.randomSample();
    return out;
  }
  // legacy_gauss: polar Box-Muller, caching the second value.
  standardNormal() {
    if (this.hasGauss) { this.hasGauss = false; return this.gauss; }
    let x1, x2, r2;
    do {
      x1 = 2.0 * this.randomSample() - 1.0;
      x2 = 2.0 * this.randomSample() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 === 0.0);
    const f = Math.sqrt(-2.0 * Math.log(r2) / r2);
    this.gauss = f * x1; this.hasGauss = true;
    return f * x2;
  }
  normal(loc, scale, size) {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) out[i] = loc + scale * this.standardNormal();
    return out;
  }
}

// One "process" worth of global RNG state, as the Python package uses the global random and np.random modules.
export function makeRng(seed = null) {
  const s = seed == null ? Math.floor(Math.random() * 2 ** 31) : seed;
  return { py: new PyRandom(s), np: new NpRandom(s), seed: s };
}

// Python repr(float): shortest round-trip digits, exponent form outside 1e-4 <= |x| < 1e16, always a '.' or 'e'.
export function pyFloatStr(x) {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const ax = Math.abs(x);
  if (ax >= 1e-4 && ax < 1e16) {
    const s = String(x);
    return /[.e]/.test(s) ? s : s + '.0';
  }
  let [m, e] = x.toExponential().split('e');
  const n = parseInt(e, 10);
  return `${m}e${n < 0 ? '-' : '+'}${String(Math.abs(n)).padStart(2, '0')}`;
}

// Python float(text) for the literals the engines produce; NaN when Python would raise ValueError.
export function pyFloatParse(text) {
  const t = text.trim();
  return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t) ? Number(t) : NaN;
}

// Python float modulo (result has the sign of the divisor).
export function pmod(x, m) {
  let r = x % m;
  if (r && (m < 0) !== (r < 0)) r += m;
  return r;
}

// colorsys.hsv_to_rgb
export function hsvToRgb(h, s, v) {
  if (s === 0.0) return [v, v, v];
  let i = Math.trunc(h * 6.0);
  const f = h * 6.0 - i, p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
  i = pmod(i, 6);
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
}
