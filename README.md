# wgpu-load-test
Examples of how different cards perform far better that others

You should be able to just `cargo run --release` in each directory to get a pretty heavy load that might kill your computer.

I repeat, this will mostly *RENDER YOUR COMPUTER useless. USE AT YOUR OWN RISK.*

* `Vertex` creates a very fine icosphere (> 20 million triangles) and tries to displace all the vertices in a heavy vertex shader. On my Radeon 5500M (MacBookPro 16") I get roughly 10fps. On my linux box with a 2080Ti it's half that. It's mostly due to register pressure (apparently, I'm yet to fully understand the issues here).
* `Physarum` is a compute pass that combines particle simulation and accumulation of traces. The algorithm is described at https://www.sagejenson.com/physarum. It's again running pretty well on my Radeon, but not at all on the RTX. *FIXED now: the buffers were not actually where they should be. This behavior is now fixed in wgpu-rs master!*
