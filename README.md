# Multiple_curved_lane_segmentation

Steps:
1. Use Yolov3 spp to find cars boundingbox, save as mask M
2. Convert raw RGB image to gray
3. Use canny to find edges
4. Find contour of the edge image
5. Apply mask M to the contour image
6. Find minimum enclosing rotated rect for all contour segments
7. Filter out those rects with poor aspect-ratio, area and area ratio to rect area 
8. Represent the rect as a line, do merging of line by angle diff and interesection result
9. Merge the 2 results of (8) and (4)
10. Perform find contour on (9) again, do polyfit to try to extend the segment
11. Draw segment with different colors to represent different lanes
