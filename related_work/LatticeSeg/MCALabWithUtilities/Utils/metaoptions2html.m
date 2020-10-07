function metaoptions2html(options,filename)
% Creates an html file from the MCALab metadata structure options.
%

f = fopen(filename,'wt');
fprintf(f,'<html>\n');
fprintf(f,'<head>\n');

fprintf(f,'<title> \n MCALab summary document \n </title>\n');

fprintf(f,'</head>\n');
fprintf(f,'<body>\n');

if isfield(options,'inputdata'),
	fprintf(f,'<table style="background-color: rgb(204, 204, 255);\n');
	fprintf(f,'text-align: left; margin-left: auto; margin-right: auto; font-style: italic; height: 57px; width: 760px;">\n');
	fprintf(f,'<tbody>\n');
    	fprintf(f,'<tr align="center">\n');
    	fprintf(f,'<td>\n');
 	fprintf(f,'<H1> %s </H1> \n',options.inputdata);
	if isfield(options,'maskdata'), fprintf(f,'<H1> %s </H1> \n',options.maskdata); end
   	fprintf(f,'</td>\n');
    	fprintf(f,'</tr>\n');
    	fprintf(f,'</tbody>\n');
	fprintf(f,'</table>\n');
end

%
if isfield(options,'mcalabver'),
	fprintf(f,'<H2> Versions and architecture </H2>\n');
	fprintf(f,'%s <br>\n',options.arch);
	fprintf(f,'%s <br>\n',options.matlabver);
	fprintf(f,'%s <br>\n',options.mcalabver);
	fprintf(f,'%s <br>\n',options.wavelabver);
	fprintf(f,'%s <br>\n',options.curvrlabver);
	fprintf(f,'%s <br>\n',options.rwtver);
end

%
if isfield(options,'algorithm'),
	fprintf(f,'<H2> Algorithm task </H2>\n');
        fprintf(f,'%s <br>\n',options.algorithm);
        fprintf(f,'%s <br>\n',options.inpaint);
end

%
if isfield(options,'dict'),
	fprintf(f,'<H2> Dictionary </H2>\n');
	fprintf(f,'%s <br>\n',options.dict);
        fprintf(f,'%s <br>\n',options.pars1);
        fprintf(f,'%s <br>\n',options.pars2);
        fprintf(f,'%s <br>\n',options.pars3);
end
%	
fprintf(f,'<H2> Algorithm parameters </H2>\n');
if isfield(options,'itermax'), fprintf(f,'%s <br>\n',options.itermax); end
if isfield(options,'epsilon'), fprintf(f,'%s <br>\n',options.epsilon); end
fprintf(f,'%s <br>\n',options.nbcomp);
if isfield(options,'tvregparam'), 
	fprintf(f,'%s <br>\n',options.tvregparam);
	fprintf(f,'%s <br>\n',options.tvcomponent);
end
fprintf(f,'%s <br>\n',options.thdtype);
fprintf(f,'%s <br>\n',options.sigma);
if isfield(options,'stopciterion'), fprintf(f,'%s <br>\n',options.stopcriterion); end
if isfield(options,'lambdamax'), 
	fprintf(f,'%s <br>\n',options.lambdamax);
	fprintf(f,'%s <br>\n',options.lambdasched);
end
if isfield(options,'lambda'), fprintf(f,'%s <br>\n',options.lambda); end
fprintf(f,'%s <br>\n',options.verbose);


fprintf(f,'</body>\n');
fprintf(f,'</html>\n');


fclose(f);
