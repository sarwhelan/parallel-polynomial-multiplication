makeall: q1_swhela2.cu q1_swhela2.cu
	nvcc q1_swhela2.cu -o q1
	nvcc q2_swhela2.cu -o q2
