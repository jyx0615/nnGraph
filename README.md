# Neural Network Sketch

A dependency-free editor for neural network architecture diagrams. Set layer names, neuron counts, activation functions, loss, and diagram width. Copy the graph or download it as SVG or PNG.

## Files

The website is in [`dist/`](dist/). `index.html` loads `styles.css` and `app.js` directly; there is no build step or `site/` folder.

## Run locally

```sh
python3 -m http.server 8765 --directory dist
```

Open <http://localhost:8765>. Your settings are saved in the browser. Clipboard copying works on localhost in browsers that support image clipboard access; SVG and PNG downloads do not need it.
