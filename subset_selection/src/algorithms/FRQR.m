function [p, R2] = FRQR(U, k, extension_method)

if nargin < 3
  tol = 1e-10;
end

[m, n] = size(U);

[~, ~, p_init] = qr(U', 'vector');
p = p_init(1:min(n, k));
p = p(:);
needed = k - numel(p);

  while needed > 0
    avail = setdiff(1:m, p);
    U_sel = U(p, :);
    U_hat = U(avail, :);

  if strcmpi(extension_method, 'mem_coh')
    R2 = 1 - abs(normr(U_hat) * normr(U_sel)');

  elseif strcmpi(extension_method, 'mem_l2')
    R2 = zeros(length(avail), length(p));
    for i = 1:length(avail)
      for j = 1:length(p)
        diff = U_hat(i,:) - U_sel(j,:);
        R2(i,j) = norm(diff, 2);
      end
    end

    else
      error('Not a valid extension method.');
    end

    [~, ~, piv] = qr(R2', 'vector');

    if ~isempty(piv)
      new_index = avail(piv(1:min(needed, length(piv))));
      p = [p; new_index(:)];
      needed = k - numel(p);
    else
      break;
    end
  end

  p = p(1:k);
end

