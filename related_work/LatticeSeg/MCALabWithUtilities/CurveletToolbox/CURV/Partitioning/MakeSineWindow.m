function w = MakeSineWindow(boxlen,alpha)

                  ix = ((-boxlen:(boxlen -1)) + .5)./boxlen;
	          w = IteratedSineWindow(ix./alpha);
	
		  
		  