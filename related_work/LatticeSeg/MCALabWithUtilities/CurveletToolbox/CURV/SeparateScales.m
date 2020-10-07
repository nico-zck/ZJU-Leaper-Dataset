function S = SeparateScales(X,L)
% SeparatesScales: Separates an image into an (orthonormal) 
%                                    series of disjoint scales.
%  Usage:
%    S =  SeparateScales(X,L);
%  Inputs:
%    X     n by n image 
%    L     scale of the coarsest coeffcients
%  Outputs:
%    S data structure which contains the FT of X at dyadic scales. 
%
% Description
%   For each l = 1, ..., J - L, 
%   S(l) is  a structure which contains scale information and the
%   FT of X tapered with a Meyer-style window which isolates
%   frequencies near a dyadic frequency subband
%   
%   C(l).scale   gives the scale associated with the lth
%                subband, e.g. S(1).scale = L-1, C(2).scale = L,
%                etc.
%
%   S(l).coeff   m by m matrix with m = = 2*2^(j+1)); matrix
%                entries are Fourier coefficients of Img at 
%                -2^(j+1) <= k1,k2 < 2^(j+1) after windowing with a 
%                bandpass Meyer window which isolates frequencies 
%                in the range 2^(j-1) <= |k| <= 2^j.  
% See Also
%   Adj_ Separatescales, Curvelet02Xform

       n = size(X,1);
       n2 = n/2;
       J = log2(n);
       deg = 3;

%
%      Create data structure
%

       S = struct('scale', L - 1, 'coeff', zeros(2*2^L));

      for j = L:(J-1), 
              S = [S struct('scale', j , 'coeff', zeros(min(2*2^(j+1),n)))];
           end
          
       F = fft2_mid0(X)/sqrt(prod(size(X)));

%      Partition at Coarse Level

       [ix,w] = CoarseMeyerWindow(L-1,deg);    
       w = [0 reverse(w(2:length(w))) w];
      
       w2 = w'*w; % Build 2D-window
       
       l = 2^L;
       lx = (n2 - l + 1):(n2 + l); 
       S(1).coeff = F(lx,lx).*w2;

%      Loop to Get Partition for  j = L, ..., J - 1;

       for j = L:(J-1),
          
	  l = min(2^(j+1),n2);
	  wlo = zeros(1,l); whi = wlo;
	 
	  [ixf, wf] = CoarseMeyerWindow(j-1,deg);
	  wlo(ixf+1)  = wf;
	  wlo = [0 reverse(wlo(2:length(wlo))) wlo]; 
	  
	  if j < J -1, 	    
	  	    [ixp, wp] = CoarseMeyerWindow(j,deg); 
		     whi(ixp+1) = wp; 
	             whi = [0 reverse(whi(2:length(whi))) whi];
	  end
	  
	  if j == J - 1, 
	            whi = ones(1,2*l); 
	  end
	  	  
	  w2 = sqrt((whi'*whi).^2 - (wlo'*wlo).^2); % Build 2D-window
	 	  
	  lx = (n2 - l + 1):(n2 + l); 
          S(j - L + 2).coeff = F(lx,lx).*w2; 	  
	end

	