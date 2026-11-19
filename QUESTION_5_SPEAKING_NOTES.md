# Question 5 presentation notes

Your section is slides 16–18 in `slides.html`. Open the presentation in a
browser and press N to show the notes on each slide. Practice once with a
timer; the assignment allows each person up to two minutes.

## Slide 16 — Complexity

“For an image with height H and width W, dynamic programming checks up to three
predecessors for every pixel. Finding one seam therefore takes O(HW) time.
Removing k vertical seams has an O(kHW) upper bound. The image gets narrower
after each removal, so the actual total is a sum over shrinking widths. The
image and energy map still require O(HW) space.”

## Slide 17 — Performance

“On a 143 by 97 version of the same image, the median time to remove one
vertical seam was 0.62 milliseconds for greedy, 2.24 milliseconds for dynamic
programming, and 727.79 milliseconds for graph cut. Dynamic programming and
graph cut found seams with the same energy. Greedy makes local choices, so it
does not guarantee the minimum total energy. These results are from three runs
on one image, and other images or machines may give different timings.”

## Slide 18 — Optimization

“The old DP search stored a full table of cumulative costs and full parent
indices. The optimized version keeps only the previous row of costs and stores
compact parent offsets so it can reconstruct the same seam. On a 500 by 700
test energy map, traced peak allocations within the seam finder fell from 2.707
to 0.376 mebibytes, about 86 percent. This excludes the input and the rest of
the program, so overall space remains O(HW). Parallel energy calculation may
help, but DP rows depend on previous rows, so any threading improvement would
need to be measured.”
