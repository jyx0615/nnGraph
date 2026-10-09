# Neural Network Sketch

A dependency-free editor for neural network architecture diagrams. Set layer names, neuron counts, activation functions, loss, and diagram width. Copy the graph or download it as SVG or PNG.

## Files

The website files—[`index.html`](index.html), [`styles.css`](styles.css), and [`app.js`](app.js)—are in the repository root. There is no build step. For GitHub Pages, choose **Deploy from a branch**, select your branch, and set the folder to **/(root)**.

## Run locally

```sh
python3 -m http.server 8765
```

Open <http://localhost:8765>. Your settings are saved in the browser. Clipboard copying works on localhost in browsers that support image clipboard access; SVG and PNG downloads do not need it.
