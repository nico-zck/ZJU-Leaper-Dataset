function w = MakeWindowIterated(boxlen,alpha)

                  ix = [(0:boxlen)/boxlen] -1/2;
	          wx = IteratedSine(ix./alpha);
	
	          wl = wx(1:boxlen);
	          wr = reverse(wx(2:(boxlen + 1)));
		  
		  w = [wl wr];
		  
		  