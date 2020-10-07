function x = Inv_UtU_CG(b,lambda,MaxIts,ErrTol,Pre,x0,mu)
% Inv_UtU_CG: Invert UtU
%  Usage:
%    x = Inv_UtU_CG(b,lambda,MaxIts,ErrTol,Pre,x0,mu);
%  Inputs:
%    b       vector of length n
%    lambda  vector of Fourier multipliers; length 2*n-1
%    MaxIts  Maximum number of CG iterations. Default 6.
%    ErrTol  Error tolerance. Default 1.e-9.
%    Pre     Boolean variable. Pre = 1 if we apply a preconditioner
%    x0      Initial guess
%    mu      Variable for use with the preconditioner
%  Outputs:
%    x       Vector of lenght n
%  Description 
%    Performs the inverse UtU. This is an approximate inverse which 
%    uses a conjugate gradient solver. Each iteration uses
%    UtU_Toeplitz which is very fast as it merely involves 2 FFTs.  
% See Also
%   GUSFT_Toelplitz

if (nargin < 2)
   error('Not enough input arguments.');
end

if (nargin < 3) | isempty(MaxIts)
   MaxIts = 6;
end

if (nargin < 4) | isempty(ErrTol)
   ErrTol = 1.e-9;
end

if (nargin < 5) | isempty(Pre)
	Pre = 0;
end

if (nargin < 6) | isempty(x0)
	x0 = b;
end

if (nargin < 7) | isempty(mu)
	mu = ones(size(b));
end

% Initialization
xk = x0;

if MaxIts > 0,
	 temp = UtU(xk,lambda);
	
	 rk=b-temp; 
	 if Pre == 0,
	   dk = rk;
	   else
	     dk=ApplyCirculantPrecond(rk,mu); 
	 end	 	 

	% Conjugate gradient iteration
	for j=1:MaxIts,
		 perr=max(abs(rk));
		 fprintf(['Iteration ',int2str(j-1),': Residual error=']);
		 fprintf('%f\n',perr);
	
		 if perr>ErrTol,			
			  temp = UtU(dk,lambda);
			  a0 = sum(conj(rk).*dk);
			  a1 = sum(conj(dk).*temp);
	
			  % Update x and residual 
			  a=a0/a1;
			  xk=xk+a*dk;		
			  rk=rk-a*temp;
			
			  % Update search direction
			  if Pre == 0,
			    b = sum(abs(rk.^2))/a0;
			    dk = rk+b*dk;
			  else
			    rkp = ApplyCirculantPrecond(rk,mu); 
			    b = sum(conj(rkp).*rk)/a0;
			    dk = rkp+b*dk;
			  end 
		 end
	end
end

x = xk;

%  References: Jonathan R. Shewchuk "An introduction to the Conjugate
%  Gradient Method Without the Agonizing Pain"
%
%  Amir Averbuch and David L. Donoho "BeamLab 200"
