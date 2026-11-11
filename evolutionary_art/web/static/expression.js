// Expression genomes: a parser + vectorised evaluator for the Python expression strings both systems generate
// (basic: "((1.2 * x) + sin((-0.4 * y)))"; enhanced: "np.sin(self.sigmoid((0.7 * x), None))", "self.flow(x, y)").
// Evaluation follows numpy element-wise arithmetic in the same operation order as the Python methods.

// --- parser --------------------------------------------------------------------------------------------------
function tokenize(s) {
  const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][\w.]*)|(\S))/y;
  const out = []; let m;
  re.lastIndex = 0;
  while (re.lastIndex < s.length && (m = re.exec(s))) {
    if (m[1] !== undefined) out.push({ t: 'num', v: Number(m[1]) });
    else if (m[2] !== undefined) out.push({ t: 'name', v: m[2] });
    else out.push({ t: 'op', v: m[3] });
  }
  if (/\S/.test(s.slice(re.lastIndex))) throw new Error('cannot parse expression');
  return out;
}

// Parse into nodes: {k:'num', v} | {k:'var', v:'x'|'y'} | {k:'none'} | {k:'neg', a} | {k:'+'|'*', a, b} | {k:'call', f, args}
export function parse(src) {
  const tk = tokenize(src); let i = 0;
  const peek = () => tk[i], eat = v => { if (!tk[i] || tk[i].v !== v) throw new Error(`expected '${v}'`); i++; };
  function primary() {
    const t = tk[i++];
    if (!t) throw new Error('unexpected end of expression');
    if (t.t === 'num') return { k: 'num', v: t.v };
    if (t.t === 'op' && t.v === '-') return { k: 'neg', a: primary() };
    if (t.t === 'op' && t.v === '+') return primary();
    if (t.t === 'op' && t.v === '(') { const e = expr(); eat(')'); return e; }
    if (t.t === 'name') {
      if (t.v === 'x' || t.v === 'y') return { k: 'var', v: t.v };
      if (t.v === 'None') return { k: 'none' };
      const f = t.v.replace(/^(np|self)\./, '');
      if (!FUNCS[f]) throw new Error(`unknown function ${t.v}`);
      eat('('); const args = [expr()];
      while (peek() && peek().v === ',') { i++; args.push(expr()); }
      eat(')');
      return { k: 'call', f, args };
    }
    throw new Error(`unexpected '${t.v}'`);
  }
  function term() { let a = primary(); while (peek() && peek().v === '*') { i++; a = { k: '*', a, b: primary() }; } return a; }
  function expr() { let a = term(); while (peek() && peek().v === '+') { i++; a = { k: '+', a, b: term() }; } return a; }
  const e = expr();
  if (i !== tk.length) throw new Error('trailing input in expression');
  return e;
}

// --- vectorised evaluation (values are numbers or Float64Arrays of one length) ---------------------------------
const map1 = (a, f) => (typeof a === 'number' ? f(a) : a.map(f));
function map2(a, b, f) {
  if (typeof a === 'number' && typeof b === 'number') return f(a, b);
  const n = typeof a === 'number' ? b.length : a.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = f(typeof a === 'number' ? a : a[i], typeof b === 'number' ? b : b[i]);
  return out;
}

// EnhancedArtisticFunction methods (styles.py), element-wise.
const sigmoid = v => 1 / (1 + Math.exp(-v));
const smoothAbs = v => Math.sqrt(v * v + 1e-6);
const flow = (x, y) => Math.sin(x + Math.cos(y)) * Math.cos(y + Math.sin(x));
const wave = (x, y) => Math.sin(3 * Math.sqrt(x * x + y * y));
function turbulence(x, y) {
  let value = 0;
  for (let i = 0; i < 3; i++) { const freq = 2 ** i; value += Math.sin(x * freq + y * freq) / freq; }
  return value;
}

export const FUNCS = {
  sin: a => map1(a[0], Math.sin), cos: a => map1(a[0], Math.cos), abs: a => map1(a[0], Math.abs),
  sigmoid: a => (a[1] != null ? map2(a[0], a[1], (x, y) => sigmoid(x + y)) : map1(a[0], sigmoid)),
  smooth_abs: a => (a[1] != null ? map2(a[0], a[1], (x, y) => Math.sqrt(x * x + y * y + 1e-6)) : map1(a[0], smoothAbs)),
  flow: a => map2(a[0], a[1], flow), wave: a => map2(a[0], a[1], wave), turbulence: a => map2(a[0], a[1], turbulence),
};

export function evaluate(node, X, Y) {
  switch (node.k) {
    case 'num': return node.v;
    case 'var': return node.v === 'x' ? X : Y;
    case 'none': return null;
    case 'neg': return map1(evaluate(node.a, X, Y), v => -v);
    case '+': return map2(evaluate(node.a, X, Y), evaluate(node.b, X, Y), (p, q) => p + q);
    case '*': return map2(evaluate(node.a, X, Y), evaluate(node.b, X, Y), (p, q) => p * q);
    case 'call': return FUNCS[node.f](node.args.map(a => evaluate(a, X, Y)));
    default: throw new Error('bad node');
  }
}

// Evaluate an expression string on grids; always returns a Float64Array of X.length values.
export function evalExpression(src, X, Y) {
  const v = evaluate(typeof src === 'string' ? parse(src) : src, X, Y);
  return typeof v === 'number' ? new Float64Array(X.length).fill(v) : v;
}

// np.linspace(-1, 1, n) and np.meshgrid: X varies along a row, Y down the columns.
export function grid(w, h) {
  const lin = n => Array.from({ length: n }, (_, i) => (n > 1 && i === n - 1 ? 1 : i * (2 / (n - 1)) + -1));
  const xs = lin(w), ys = lin(h), X = new Float64Array(w * h), Y = new Float64Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) { X[r * w + c] = xs[c]; Y[r * w + c] = ys[r]; }
  return { X, Y };
}

// (r - min) / (max - min + eps)
export function normalize(a, eps) {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const d = hi - lo + eps, out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - lo) / d;
  return out;
}

export const allFinite = a => a.every(Number.isFinite);
