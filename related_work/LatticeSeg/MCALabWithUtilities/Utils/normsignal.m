function xn=normsignal(x,mn,mx)

xmn=min(x(:));
xmx=max(x(:));

xn=(mx-mn)*(x-xmn)/(xmx-xmn)+mn;
