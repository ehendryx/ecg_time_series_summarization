function p = RRQR(U, k, tol, extension_method)

if nargin < 3
  tol = 1e-10;
end

[n, m] = size(U);

[~, ~, p_init] = qr(U', 'vector');
p = p_init(1:min(m, k));
p = p(:);  % ensure column vector
needed = k - numel(p);

while needed > 0
  A = setdiff(1:n, p);
  U_A = U(A, :);           % candidate rows
  U_p = U(p, :);               % selected rows

  if strcmpi(extension_method, 'mem_coh')
    % coherence-based (cosine distance)
    w = 1 - max(abs(normr(U_A) * normr(U_p)'), [], 2);

  elseif strcmpi(extension_method, 'mem_l2')
    % L2-norm distance to closest selected row
    diffs = zeros(size(U_A,1), size(U_p,1));
    for j = 1:size(U_p,1)
      diffs(:, j) = vecnorm(U_A - repmat(U_p(j,:), size(U_A,1), 1), 2, 2);
    end
    w = min(diffs, [], 2);
    w = w / max(w);  % normalize

  else
    disp('Not a valid extension.');
  end

  % apply memory-informed weighting and QR to find next pivots
  % (U_weighted contains residuals scaled based on similarity to previously selected rows)
  U_w = U_A .* w;
  [~, R, piv] = qr(U_w', 'vector');

  % optional: truncate pivots based on residual strength
  d = abs(diag(R));
  cut = find(d < tol, 1);
  if ~isempty(cut)
    piv = piv(1:cut - 1);
  end

  % pick the top pivot
  if ~isempty(piv)
    new_index = A(piv);
    p = [p; new_index(:)];
    needed = k - numel(p);
  else
    break; % no viable new indices
  end
end

% trim in case of overshoot
p = p(1:k);
end

