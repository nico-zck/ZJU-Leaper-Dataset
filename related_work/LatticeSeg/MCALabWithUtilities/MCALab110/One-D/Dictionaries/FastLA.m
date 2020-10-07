function c = FastLA(x, NameOfDict, par1, par2, par3)
% FastLA -- the ANALYSIS operator for a dictionary:
%			c = \Phi^T * x
%  Usage:
%	c = FastA(x, NameOfDict, par1, par2, par3)
%  Inputs:
% 	x		the signal, a column vector
%	NameOfDict	name of the dictionary
%	par1,par2,par3	the parameters of the dictionary
%
%	Use 'help dictionary' for dictionary objects: NameOfDict,par1,par2,par3
%  Outputs:
%	c		the coefs, structure array
%  See Also:
%	FastAA, FastS, FastSS, MakeList
%

NumberOfDicts = LengthList(NameOfDict);
if NumberOfDicts == 1,
	c{1} = eval(['Fast' NameOfDict 'Analysis(x, par1, par2, par3)']);
else
	c = {};
	for i = 1:NumberOfDicts,
		NAME = NthList(NameOfDict, i);
		PAR1 = NthList(par1, i);
		PAR2 = NthList(par2, i);
		PAR3 = NthList(par3, i);
		c{i} = eval(['Fast' NAME, 'Analysis(x, PAR1, PAR2, PAR3)']);
	end
end
