% MPWork22 -- Matching-Pursuit, Wavelet Packet Tour on Four Complex Signals
%
%     	Sorrows
% 		Riemann
%		Seismic
%		Tweet
%
	n   = 1024;
	qmf = MakeONFilter('Coiflet',3);
%
	compressfig  = figure;
	equivratefig = figure;
	equivdimfig  = figure;
	figarray = zeros(4,1);
	brightarray = [-.8 -.8 0 0];
	axarray = [ 0 0 0 0; ...
	            0 0 0 0; ...
				0 0 0 0; ...
				0 1 .3 .7];
%
	for i=1:4,

		if i==1,
			SigName = 'WernerSorrows';
			hs = MakeSignal(SigName,n);
			SigName = 'Sorrows';
		elseif i==2,
			SigName = 'Riemann';
			hs = MakeSignal(SigName,n);
			hs = hs ./ norm(hs) .* sqrt(1000);
		elseif i==3,
			SigName = 'Seismic';
			hs = ReadSignal('Seismic')';
			hs = hs ./ norm(hs) .* sqrt(1000);
		elseif i==4,
			SigName = 'Tweet';
			hs = ReadSignal('Tweet');
			hs = hs ./ norm(hs) .* sqrt(8192);
		end			
%
		figarray(i) = figure;
		[mpatoms,mpresid] = WPPursuitTour('P',hs,10,qmf,300,SigName);
		subplot(2,2,3)
		axis([0 500 0 10])
		
		if brightarray(i) ~= 0,
			subplot(2,2,4)
			brighten(brightarray(i));
		end
	
		if sum(abs(axarray(i,:)')) ~=0,
		    subplot(2,2,4)
			axis(axarray(i,:));
		end
		
		energy = norm(hs).^2;
		mpcnums = [energy ; energy - cumsum(mpatoms(:,1).^2)];

		figure(compressfig);
		subplot(2,2,i)
		PlotSemiCompressNum(mpcnums,'MPWP', SigName);

		figure(equivratefig);
		subplot(2,2,i)
		PlotEquivRate(mpcnums,'WP', SigName)

		figure(equivdimfig);
		subplot(2,2,i)
		PlotEquivDimension(mpcnums,'WP', SigName)

	end
%
	%ifprint(compressfig, 'mp_fig225.ps')
	%ifprint(equivratefig,'mp_fig226.ps')
	%ifprint(equivratefig,'mp_fig227.ps')
	
	for i=1:4,
		ifprint(figarray(i),['mp_fig22' int2str(i) '.ps']);
	end

% Revision History

% 11/08/05   A.M.    cwarb.mat is replaced by tweet.
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:43 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
