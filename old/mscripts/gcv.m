function mu=gcv(A,L,b,opt_upper_bound)

% Allow us the mess with the upper bound of fminbound
if nargin < 4
    opt_upper_bound = 200; % Default value for z
end

[U,~,~,S,La] = gsvd(A,L,"econ");
bhat=U'*b;
l=diag(La);
s=diag(S);
mu=fminbnd(@gcv_funct,0,opt_upper_bound,[],s,l,bhat);
end

function G=gcv_funct(mu,s,l,bhat)
num=(l.^2.*bhat./(s.^2+mu*l.^2)).^2;
num=sum(num);
den=(l.^2./(s.^2+mu*l.^2));
den=sum(den)^2;
G=num/den;
end