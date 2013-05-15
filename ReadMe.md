ImageConvolution application performs convolve operator on give image.

Qt is used to design the front end of the application.

CUDA handles the convolution operation.

	To peform convolution on a given image, carry out the below steps.
	1) Click File->Open to open an image.
	2) Click Convolve->Set Kernel to set the kernel for convolve operator
	3) Click Convolve->Pad Image to add padding pixels along the border of
	   the input image to handle convolution at edges without any boundary checking.
	4) Click Convolve->Apply Kernel to convolve the input image with supplied kernel.
	5) Once step 4 is carried out, the resultant convolved image will be shown.
