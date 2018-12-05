# Rotate RoI Align Pooling and Rotate Position Sensitive RoI Align Pooling in Caffe

This repository contains codes of two popular Rotate RoI Pooling operation in Caffe, modified from the regular implementation from [codes](https://github.com/zengarden/light_head_rcnn).

## ##
## Usage

- You need firstly perform conversion from point-from (x1, y1, x2, y2, x3, y3, x4, y4) of the quadrilaterals to rotate bouding boxes in (x, y, w, h, angle). Please consider the following pseudo codes:

	```cpp
	pt1 = (x1, y1), pt2 = (x2, y2), pt3 = (x3, y3), pt4 = (x4, y4)

	edge1 = sqrt((pt1[0] - pt2[0]) ^ 2 + (pt1[1] - pt2[1]) ^ 2)
	edge2 = sqrt((pt2[0] - pt3[0]) ^ 2 + (pt2[1] - pt3[1]) ^ 2)

	width = min(edge1, edge2)
	height = max(edge1, edge2)
	
	if edge1 > edge2:
		angle = 90.0 if pt1[0] - pt2[0] == 0 else -arctan((pt1[1] - pt2[1]) / (pt1[0] - pt2[0])) / pi * 180
	elif edge2 >= edge1:
		angle = 90.0 if pt2[0] - pt3[0] == 0 else -arctan((pt2[1] - pt3[1]) / (pt2[0] - pt3[0])) / pi * 180
	while angle < -45.0:
		angle = angle + 180

	x = (pt1[0] + pt3[0]) / 2.0
	y = (pt1[1] + pt3[1]) / 2.0
	```

- Now you can add these operation in caffe.proto and rebuild the library, then call with params similar as the regular ones:


## ##
Apache License, Version 2.0