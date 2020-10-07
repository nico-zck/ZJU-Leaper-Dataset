function X = Adj_SeparateAngles(R,deep,IsImageReal)
% Adj_SeparateAngles -- Adjoint of the frequency angular partioning 
% Usage:
%   X = Adj_SeparateAngles(R,deep)
% Inputs:
%   R         Array of smoothly localized Fourier samples 
%   deep      Depth of the angular partition; number of angular
%             sectors per cardinal point (NESW) = 2^deep
% Outputs
%   X         squared array which we interpret as FT at a given 
%             scale
% See Also
%   SeparateAngles, AtA, AtA_Toeplitz

if nargin < 3,
  IsImageReal = 0;
end

	nn  = size(R);
	n = 4*nn(3);
	n2 = n/2;	
        boxlen = n/2^deep;
	boxcnt = 2^deep;	
		
	X = zeros(n);
	XX = zeros(n);
	   	
	[ix,w] = DetailMeyerWindow([n2/4 n2/2],3);
	lx = reverse(-ix);
	 
	m      =  0:(boxcnt - 1);
        ym     =  (m-boxcnt/2).*boxlen + boxlen/2;
	slope  =  -ym./n2;		 
		
% Columns

	for r = 1:(n2/2),	    
	    k = lx(r); t = n2 + k + 1;
	    shift  =  ym + slope.*(k+n2);
	    alpha =  -k./n2;
	    w = MakeSineWindow(boxlen,alpha);	
            
	    y = squeeze(R(1,:,r,:)).';
	    y =  Adj_Evaluate_FT(y(:),shift,boxlen,0,w);
	    
	    X(:,t)  = y;
		
	    rsym = n2/2+1-r; t = n2 - k + 1; 
	    if (IsImageReal)
	      X(:,t) = conj(y);
	    else
	      shift = -shift + 1;
	      y = squeeze(R(2,:,rsym,:)).';
	      X(:,t) = Adj_Evaluate_FT(y(:),shift,boxlen,0,w);
	    end
	    
	end
	
	
	X = fft_mid0(X)./sqrt(n);
	
% Rows

	for r = 1:(n2/2),	      
	    k = lx(r);  t = n2 + k + 1;
	    shift  =  ym + slope.*(k+n2);
	    alpha =  -k./n2;
	    w = MakeSineWindow(boxlen,alpha);	
	    
	    y = squeeze(R(3,:,r,:)).';
	    y = Adj_Evaluate_FT(y(:),shift,boxlen,0,w);
	    
	    XX(:,t)  = y; 

            rsym = n2/2+1-r; t = n2 - k + 1; 
	    if (IsImageReal)
	      XX(:,t) = conj(y);
	      else
		shift = -shift + 1;
		y = squeeze(R(4,:,rsym,:)).';
	        XX(:,t) = Adj_Evaluate_FT(y(:),shift,boxlen,0,w);
	    end
	    
	end	
 
	XX = fft_mid0(XX).'/sqrt(n);
	X = X + XX;

	

		  		  
	
	
	
		
	
	

