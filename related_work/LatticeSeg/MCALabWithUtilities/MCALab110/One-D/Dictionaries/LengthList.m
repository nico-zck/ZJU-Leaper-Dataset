function n = LengthList(list)
% LengthList -- length of a list data structure
%  Usage
%	n = LengthList(list)
%  Inputs
%	list	the list data structure
%  Outputs
%	n	the length of list
%  See Also
%	MakeList, NthList
%

if isstr(list)
	n = sum(list == ' ');
else
	n = sum(list == inf);
end

if n == 0,
	n  = 1;
end