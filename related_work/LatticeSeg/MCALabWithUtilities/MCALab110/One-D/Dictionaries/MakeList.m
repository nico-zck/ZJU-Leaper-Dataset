function list = MakeList(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
% MakeList -- Make a list structure of strings/vectors
%  Usage
%	list = MakeList(arg1[, arg2, arg3, arg4, arg5, arg6, arg7, arg8])
%  Inputs
%	arg1 ...	the items you want to put into a list, they can
%			either be all strings, or, all vector of numbers
%  Outputs
%	list		the list data structure
%  See Also
%	LengthList, NthList
%

% strings or numbers
if nargin == 1,
	list = arg1;
else
	list = [];
	if isstr(arg1)
		for i = 1:nargin,
			cmmdstr = sprintf('arg%g', i);
			list = [list eval(cmmdstr) ' '];
		end
	else
		for i = 1:nargin,
			cmmdstr = sprintf('arg%g', i);
			list = [list eval(cmmdstr) inf];
		end
	end
end


	
