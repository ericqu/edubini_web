#!/bin/sh
# Build the WebAssembly engine and copy it next to the web app.
set -e
cd "$(dirname "$0")/engine"
cargo build --release --target wasm32-unknown-unknown --lib
cp target/wasm32-unknown-unknown/release/microscope_engine.wasm ../engine.wasm
ls -l ../engine.wasm
