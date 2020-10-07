function C = SymmetrizePair(D);

         nn = size(D); C = zeros(nn);

	 k1 = 0:(nn(3) - 1);
	 k2 = 0:(nn(4) - 1);
	 Omega = zeros(1,1,nn(3),nn(4));
	 Omega(1,1,:,:) = exp(i*2*pi*k1.'./nn(3)) * exp(i*2*pi*k2./nn(4));
	 
	 for j = [1 3],
	   for m = 1:nn(2),
	     C(j,m,:,:) = D(j,m,:,:) + D(j+1,m,:,:).*Omega;
             C(j+1,m,:,:) = (D(j,m,:,:) - D(j+1,m,:,:).*Omega)/i;
	   end
	 end
	 
	 C = C./sqrt(2);
	 
	 
         
         
         
	 
	 
	     
	 