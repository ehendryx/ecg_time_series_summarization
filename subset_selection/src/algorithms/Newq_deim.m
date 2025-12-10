function [ S, M] = Newq_deim(U, limit) ;
% Input : U n−by−m with orthonormal columns
% Output : S selection of m row indices with guaranteed upper bound
%          norm(inv(U(S,:))) <= sqrt(n−m+1) * O(2ˆm).
%        : M the matrix U*inv(U(S,:));
%          The Q−DEIM projection of an n−by−1 vector f is M*f(S).
% Coded by Zlatko Drmac, April 2015.
[n,m] = size(U) ;
m = min([m,limit]);
if nargout == 1
  [~, R, P] = qr(U','vector') ; S = P(1:m) ;
else
  [Q,R,P] = qr(U','vector') ; S = P(1:m) ;
   M = [eye(m) ; (R(:,1:m)\R(:,m+1:n))'] ;
   Pinverse(P) = 1 : n ; M = M(Pinverse,:) ;
end
end
