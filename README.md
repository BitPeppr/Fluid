# Spume

A python terminal-based fluid simulator with ASCII visualisations and minimal dependencies (no rendering dependencies).

## Features

- Minimal ascii, completely cli-based visualisation
- Minimal external dependencies
- Interactive; click to disperse, WASD to move spawn point
- Highly extensible; diverse CLI flags to customise physics and visualisation

## Installation

```bash
pip install spume
spume -h # Look at possible options
```

## Usage

WASD to move the spawner around, left mouse click (with drag) to disperse fluid, q to exit.

You can also use the CLI flags to customise the simulation, e.g. spawner size, sim maxiter / tolerance (depending on how beefy your laptop is), etc.

Enjoy!

## Contributing

> What I really need right now is good names; I wanted plume, but that was taken on pypi; so I went for spume, but it doesn't seem as poetic. Any ideas?
