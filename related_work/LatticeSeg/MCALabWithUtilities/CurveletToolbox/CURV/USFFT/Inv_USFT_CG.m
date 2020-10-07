function x = Inv_USFT_CG(y,shift,boxlen,Pre,w)

if nargin < 5,
  w = ones(1,2*boxlen);
end

if nargin < 4,
  Pre = 0;
end

        ty = Adj_USFT_simple(y,shift,boxlen,0,w);

% Preparation

	n = length(ty); 
	col = Adj_USFT_simple(ones(2*n,1),shift,boxlen,1,w)./sqrt(n);
	row =  conj(col); 	
	extended.col = [col; reverse(row(2:n))];  
	
	lambda = ifft(extended.col).*length(extended.col);	
	
% Preconditioner
         
      eps = 1.e-6;
      if Pre == 1
	method = 'Strang';
        pre.col = MakeCirculantPrecond(col,method);
  	mu = ifft(pre.col).*n + eps;
      else 
	mu = [];
      end
      
	
 	 	  	
% Set parameters	
        
        if Pre == 0
       	  b = ty; 
	else
	  b = ApplyCirculantPrecond(ty,lambda,mu);
	end	
	maxits = 10;
       	x = Inv_UtU_CG(b,lambda,mu,b,Pre,maxits);
	
	
	

		
	
	
	
		  		  
	
	
	
		
	
	

