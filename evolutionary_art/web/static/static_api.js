// Static-site data layer: answers api('/...') from the JSON files exported by scripts/build_static.py (every GET in
// STATIC_ENDPOINTS of evolutionary_art/web/app.py). Evolution itself runs in the browser (basic_engine.js,
// enhanced_engine.js); only POST /evolve, which runs the Python package, needs the local server.
const files = new Map();

export async function staticApi(path, body) {
  const rel = path.split('?')[0].replace(/^\/+/, '');
  if (body) throw new Error('Running the Python package needs the local app (python run_web.py)');
  if (!files.has(rel)) {
    files.set(rel, fetch('api/' + rel + '.json').then(r => {
      if (!r.ok) throw new Error(`${r.status} ${rel}`);
      return r.json();
    }));
  }
  return files.get(rel);
}
