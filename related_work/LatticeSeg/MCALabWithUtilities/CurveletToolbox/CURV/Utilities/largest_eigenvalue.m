W = SetScaleToZero(n);
q = randn(n).*W;
q = q./norm(q,'fro');
z = AtA_Toeplitz(q,Lambda).*W;

niter = 50;

for k = 1:niter,  
   q = z./norm(z,'fro');
   z = AtA_Toeplitz(q,Lambda).*W;
   lambda = sum(sum(conj(q).*z))
end


