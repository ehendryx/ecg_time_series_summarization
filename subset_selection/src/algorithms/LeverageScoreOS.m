function [p] = LeverageScoreOS(U,k,pvalue)
  %% U is from svd
  %% e is number of right singular vectors used in computing leverage scores
  %% k is max size of the selected indices
  %% p is the column subset selection

  [m,e] = size(U);

  r = zeros(m,1);
  for j = 1:m
    r(j) = r(j) + norm(U(j,:)', pvalue)^2;
  end
  r = (1/e)*r;
  [s,i] = sort(r, 'descend');
##  limit = min([e,k]);
  p = i(1:k,1);
