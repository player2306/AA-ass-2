# Question 5: Analysis and performance

This is a draft for the group report. It covers the current implementation and
should be reviewed alongside Members 2, 3 and 4's sections. The assignment's
Question 5(a) also refers to Question 2, which the group allocated to Member 2;
the short answers below can be reconciled with that member's detailed proofs.

## Question 2: theory

For an image with height H and width W > 1, there are at least 2^H vertical
seams: choose either of the first two columns independently in every row. Every
such choice is valid because adjacent rows then differ by at most one column.
Thus the number of possible seams grows at least exponentially with H.

Let E(r,c) be the energy of pixel (r,c). The minimum energy of a seam ending at
that pixel is M(0,c) = E(0,c) for the first row and

    M(r,c) = E(r,c) + min M(r-1,p), for valid p in {c-1,c,c+1}.

The answer is the smallest value in the last row, followed by backtracking its
parent choices. The recurrence works because every valid seam ending at (r,c)
must arrive from one of those three predecessors; choosing the cheapest optimal
prefix minimizes the whole seam.

A fixed-size Sobel filter does constant work per pixel, so a direct convolution
is already Θ(HW) for a full image. Dividing the image into subregions does not
change that sequential asymptotic cost and introduces boundaries and overhead.
Independent regions can be computed in parallel with the needed halo pixels,
but speedup depends on available cores, memory bandwidth, and overhead. For a
different or larger kernel, other convolution methods may have different costs.

## Time and space complexity

H and W are the initial height and width, k is the number of vertical seams,
and the image has a constant number of channels. The usual output is assumed:
visualization frames are not retained. Horizontal carving uses the same method
on a transposed image, exchanging H and W.

| Method | Seam search for one fixed energy map | Other work and space |
| --- | --- | --- |
| DP | Θ(HW) time: each pixel considers at most three predecessors. | The optimized search stores one byte per pixel for parents plus O(W) rolling costs and O(H) for the returned seam. Image and energy arrays still use Θ(HW) space. |
| Single-start greedy | O(W + H) time to choose the initial pixel and follow one path; O(H) for the returned seam. | Calculating energy and removing pixels still take Θ(HW) per seam in this implementation; image and energy arrays use Θ(HW) space. |
| Graph cut | The graph has V = HW + 2 vertices and E = Θ(HW) edges. The default general Dinic bound is O(V²E), or O((HW)³), per solve; Edmonds–Karp has O(VE²), the same coarse bound here. | Graph storage is Θ(V + E) = Θ(HW); building the graph and pixel removal add work. The bounds are worst-case, not measured predictions. |

For the DP and single-start greedy implementations, removing k vertical seams
takes Θ(H × sum(W-i for i = 0,...,k-1)) = Θ(H(kW - k(k-1)/2)) for k ≥ 1,
assuming the usual fixed-channel operations. It is also O(kHW). Although the DP
code updates Sobel values only in a band around the seam, it still handles
full-size arrays in other steps, including grayscale conversion, DP, and
deletion. Returned seam histories use O(kH) additional space; retaining step
frames can use up to O(kHW) more. Graph cut rebuilds and solves a shrinking graph
for each seam, so its total depends on the actual solver and input.

## Comparison and quality

On the same energy map, DP minimizes total seam energy. Graph cut with backward
energy also minimizes the corresponding seam cost; ties can produce different
paths, and paths may diverge after repeated removals. Greedy selects each next
pixel locally, which is faster as a seam search but does not guarantee a global
minimum. A smaller calculated seam cost is not by itself proof of better visual
quality: compare visible distortions and subject preservation in the output
images as well. Forward-energy graph cut uses a different cost, so its energy
numbers need a common evaluation measure before direct comparison.

In the existing [DP](output/q3_dp_n100.png) and
[greedy](output/q3_greedy_n100.png) images after 100 vertical removals, both
preserve the tower and person, but their placement differs: the tower is farther
left in the greedy result, indicating that the methods removed different parts
of the image. The visual comparison alone does not establish a universally
better method. The existing small [DP](output/q4_dp_n20.png) and
[graph-cut](output/q4_graphcut_n20.png) results after 20 removals look very
similar at the available resolution; examine larger examples before making a
stronger claim about visual differences.

### Measured running time

The benchmark used `Broadway_tower_edit.jpg`, resized to the same dimensions
for all three algorithms. It measured wall-clock time for a complete `run()`:
energy calculation, seam search, deletion and bookkeeping. Loading and resizing
were excluded, and visualization was disabled. Graph cut used Dinic with
backward Sobel energy; greedy used the default single-start method. The reported
time is the median of three runs after a warm-up, with method order shuffled and
OpenCV limited to one thread.

| Image (width × height) | Vertical seams | Greedy (ms) | DP (ms) | Graph cut (ms) |
| --- | ---: | ---: | ---: | ---: |
| 43 × 29 | 1 | 0.225 | 0.767 | 33.823 |
| 71 × 48 | 1 | 0.321 | 1.151 | 91.194 |
| 107 × 73 | 1 | 0.531 | 1.805 | 338.284 |
| 143 × 97 | 1 | 0.617 | 2.235 | 727.785 |
| 71 × 48 | 5 | 1.425 | 4.921 | 473.964 |

On the 143 × 97 image, the first seam's Sobel energy was 1,854 for both DP and
graph cut and 4,778 for greedy. This illustrates the cost difference for this
input, not a visual-quality score. The timing graphs use a logarithmic vertical
axis because the graph-cut times are much larger:

- [Time versus image size](output/q5_benchmark/time_vs_image_size.png)
- [Time versus number of seams](output/q5_benchmark/time_vs_seam_count.png)
- [All measurements and recorded environment](output/q5_benchmark/)

These results describe this implementation and test machine for one source
image, small scaled sizes, and three repetitions. They do not establish
universal speed ratios or how visual quality compares on other images. Refer
to Members 3 and 4's result images and discuss visible artifacts separately.
Reproduce with `python benchmark_q5.py --repeats 3` after installing the project
requirements; inputs and output path can be changed using `--help`.

## Improvement: less DP memory

The previous seam search retained an H × W float32 cumulative-cost table and an
H × W int32 table of absolute parent indices, approximately 8HW bytes combined.
The revised search retains only the preceding cumulative-cost row and stores a
signed one-byte parent offset (-1, 0 or 1) for each pixel. These persistent search
arrays need approximately HW + O(W) bytes, plus the O(H) seam output and
temporary row arrays. It keeps parent offsets so it can reconstruct the exact
seam and preserve the previous tie-breaking order. The whole program remains
Θ(HW) space because it also needs the image, energy map, and other arrays; this
is not a claim that total program memory falls by seven eighths.

On one synthetic 500 × 700 float32 energy map, the seam-search benchmark
produced the same seam before and after the change. Across seven timing runs,
median search time was 16.74 ms before and 15.18 ms after. A separate
`tracemalloc` measurement showed peak traced allocations of 2.707 MiB before
and 0.376 MiB after, about 86% less for the measured finder allocations. The
input energy map and the rest of the application were excluded. These are
results for one map and machine, not a claim about end-to-end memory or a
guaranteed speedup. See the [before-and-after graph](output/q5_benchmark/optimization_comparison.png).
Reproduce with `python benchmark_q5_optimization.py --repeats 7`.

Further possibilities include profiling and reducing copies, or changing pixel
deletion to reduce Python overhead. Validate output and measure speed and memory
before claiming an improvement. Energy computation and independent pixels in a
DP row can potentially run in parallel, but each row must wait for the preceding
row, and each seam removal must finish before the next image and energy map are
known. OpenCV/NumPy may already use native parallelism; Python thread overhead,
synchronization and memory bandwidth can cancel any gains. No threading speedup
is claimed without a controlled measurement.

## Validation and presentation notes

The rolling implementation was compared with the original on 120 random,
tie-heavy energy maps, including one-row and one-column cases. The existing
greedy/DP and graph-cut suites were also run. Run the checks from the project
root after installing requirements:

```text
python -m tests.test_dp_optimization
python -m tests.test_greedy
python -m tests.test_graphcut
```

For a short presentation, explain the O(HW) DP search, compare the measured
times in the two graphs, then show how rolling costs and compact parent offsets
reduce seam-search memory while retaining the path. State that the graph uses a
logarithmic time axis and that the measurements use one image. The team should
include this AI assistance and the main prompts in its required AI declaration,
and each member should understand the submitted work for the project quiz.
