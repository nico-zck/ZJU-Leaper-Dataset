function x = FastLS(c, NameOfDict, par1, par2, par3)
% FastLS -- the SYNTHESIS operator for a dictionary:
%			x = \Phi * c
%  Usage:
%	x = FastS(c, n, NameOfDict, par1, par2, par3)
%  Inputs:
% 	c		the coefs, a structure array
%	NameOfDict	name of the dictionary
%	par1,par2,par3	the parameters of the dictionary
%
%	Use 'help dictionary' for dictionary objects: NameOfDict,par1,par2,par3
%  Outputs:
%	x		the synthesized signal, a column vector
%  See Also:
%	FastA, FastAA, FastSS, MakeList
%

NumberOfDicts = LengthList(NameOfDict);
if NumberOfDicts == 1,
	C = c{1};
	cmmdstr = ['Fast' NameOfDict 'Synthesis(C, par1, par2, par3)'];
	x = eval(cmmdstr);
else
	x = 0;
	for i = 1:NumberOfDicts,
                NAME = NthList(NameOfDict, i);
                PAR1 = NthList(par1, i);
                PAR2 = NthList(par2, i);
                PAR3 = NthList(par3, i);
		C = c{i};
                x = x +  eval(['Fast' NAME, 'Synthesis(C, PAR1, PAR2, PAR3)']);
	end
end

