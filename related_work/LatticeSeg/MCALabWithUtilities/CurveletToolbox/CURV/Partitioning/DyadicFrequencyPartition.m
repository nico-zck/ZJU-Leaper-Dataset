function y = DyadicFrequencyPartition(xhat, L)

        n = length(xhat);
	J = log2(n);
	y = zeros(1,2*n);
% 
%  Compute Partition at Coarse Level.
%

	[index, window] = CoarseIteratedSine(L-1);
	left_index = reverse(n/2 + 1 - index);
	left = xhat(left_index).* reverse(window);
	right_index = n/2 + index;
	right = xhat(right_index).* window;
	y(1:2^(L+1)) = [left right];
	
%
%  Loop to Get Partition for  j = L - 1, ..., J - 3.
%
	for j = L-1:(J-3),
	  dyadic_points = [2^j 2^(j+1)];
	  [index, window] = DetailIteratedSine(dyadic_points); 
	  left_index = reverse(n/2 + 1 - index);
	  left = xhat(left_index).* reverse(window);
	  right_index = n/2 + index;
	  right = xhat(right_index).* window; 
	  y((2^(j+2)+1):(2^(j+3))) = [left right]; 
	end

%
%  Finest Subband (for j = J - 2).
%
        
        j = J - 2;
        [index, window] = FineIteratedSine(j);
	left_index = reverse(n/2 + 1 - index);
	left = xhat(left_index).* reverse(window);
	right_index = n/2 + index;
	right = xhat(right_index).* window; 
	y((2^(j+2)+1):(2^(j+3))) = [left right]; 
	

	
	
	
	
	









