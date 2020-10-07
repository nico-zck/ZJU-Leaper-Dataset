function y = GoodPartition(x,j)

	n  = length(x); 
	deep   = j;
	boxlen = n/2^j;
	boxcnt = 2^j;
	y = zeros(1,2*n); 		%Storage for tapered vector

%       Preparation

	m  = boxlen;
	ix = [(0:m)/m] -1/2; 
	wx = IteratedSine(ix);
	
	wl = wx(1:m);
	wr = reverse(wx(2:(m+1)));
	
%
%       Tapering 
%
	
	for bx = 0:(boxcnt-1),
	  
	   [lox,hix] = dyadbounds(deep,bx,n);
	   [l2y,h2y] = dyadbounds(deep,bx,2*n);
	   
	   if bx > 0 & bx < (boxcnt-1),	     
	            ilx = [lox:hix] - m/2;
	            irx = [lox:hix] + m/2;
		      
	            ily = l2y:(l2y+(m-1));
	            iry = (l2y+m):h2y;
	   
	            y(ily) = wl.* x(ilx);
	            y(iry) = wr.* x(irx);
	   end 
	   
	   if bx == 0;  % left tapering
	            ilx = lox:(lox+m/2 - 1);
	            irx =  [lox:hix] + m/2;
		    
		    ily = (l2y+m/2):(l2y+(m-1));
		    iry = (l2y+m):h2y;
 	 
	            y(ily)    = x(ilx);
	            y(iry)    =  x(irx).*wr;
	   end
	   
	   if bx == (boxcnt - 1) % right tapering
	                ilx = [lox:hix] - m/2;
	      	        irx = (hix - m/2 + 1):hix;
	  	   
	                ily = l2y:(l2y+(m-1));
	                iry = (l2y+m):(h2y-m/2);
		                
			y(ily) = wl.* x(ilx);
                        y(iry) = x(irx);
	     end
	end
	