clear variables
%% System Dimension
nx = 4;
nu = 1;
nd = 2;
ny = 1;

%% Random System
x0 = rand(4,1)*10 - 5;      % Random initial state
e = eye(4);                 % Identity matrix
A = rand(4,4)*4 - 2; A = A./(max(abs(eig(A))));  % Normolized A
B = [0.2; 0.1; 0.3; 0.1];
Bd = [0.1 0.15; 0.2 0.1; 0.1 0.3; 0.1 -0.1];
Cy = [1 1 1 1];
Dy = [0];
Dyd = [0.1 -0.2];


%% Horizons
Nc = 10;
Ne = 4;
N_seq = 30;

%% Disturbance
d_range_half = 0.3;
d_upper = d_range_half * ones(2, 1);
d_lower = -d_upper;
d = rand(nd, N_seq)*d_range_half*2 - d_range_half;
%% Cost Weight
Q = eye(nx);
R = eye(nu);
PN = eye(nx);

%% System Constraints
Cf = [eye(nx);zeros(nu, nx);-eye(nx);zeros(nu, nx)];
Cf_hat = [eye(nx); -eye(nx)];
Df = [zeros(nx, nu); eye(nu);zeros(nx, nu);-eye(nu)];
Dfd = zeros(2*(nx+nu), nd);

Cz = [Q^(1/2); zeros(nu,nx)];
Cz_hat = PN^(1/2);
Dz = [zeros(nx, nu); R^(1/2)];
Dzd = zeros(nx + nu, nd);
z_bar = zeros(nx + nu, 1);

xh = 2000* ones(nx, 1);xl = -xh;
uh = 1000* ones(nu, 1);ul = -uh;
xh_N = xh;xl_N = -xh_N;
fk = [xh;uh;-xl;-ul];
f_bar = [kron(ones(Nc, 1), fk);xh_N;-xl_N];

%% Stacked System Implementation
[A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Nc);
[Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
[Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
z_bold_bar = zeros(size(Cz_bold, 1), 1);
d_pred_lower = kron(ones(Nc+1, 1), d_lower);
d_pred_upper = kron(ones(Nc+1, 1), d_upper);

%% Estimation parameters
Estimation.len = 0;
%% Iterations
x = zeros(4, N_seq);
x_lower = ones(4, N_seq) * -inf;
x_upper = ones(4, N_seq) * inf;
x_lower(:,1) = -5;
x_upper(:,1) = 5;
x(:,1) = x0;
% Output sequence
y = zeros(ny, N_seq);
% Input sequence
u_out = zeros(nu, N_seq);
% Cost Sequence
cost = zeros(1, N_seq);
for k = 1:N_seq
    dk = d(:,k);
    y(:, k) = Cy * x(:,k) + Dyd * dk;
    %% Bounds Update
    if(k == 1)
        % Special case, only for x0
        y_est = y(:, 1);
        y_est = y_est(:);
        y_tilde = y_est;
        d_est_lower = d_lower;
        d_est_upper = d_upper;
        Cy_bold = Cy;
        Dyd_bold = Dyd;
        % For each entry
        for i = 1:nx
            A_new  = e(i, :);
            % Upper bounds
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variable g(1);
            variable X(nx, nx) diagonal semidefinite;
            variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
            variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
            % LMI
            L11 = g+y_tilde'*Y*y_tilde+x_upper(:,1)'*X*x_lower(:,1)+d_est_upper'*D*d_est_lower;
            L21 = -A_new'/2 - X*(x_upper(:, 1) + x_lower(:,1))/2 - Cy_bold'*Y*y_tilde;
            L22 = X + Cy_bold'*Y*Cy_bold;
            L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
            L32 = Dyd_bold'*Y*Cy_bold;
            L33 = D + Dyd_bold'*Y*Dyd_bold;
            L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
            minimize(g)
            subject to
            L >= 0
            cvx_end
            x_upper(i, 1) = min(x_upper(i,1), g);
            % Lower Bounds
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variable g(1);
            variable X(nx, nx) diagonal semidefinite;
            variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
            variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
            % LMI
            L11 = -g + y_tilde'*Y*y_tilde + x_upper(:,1)'*X*x_lower(:,1) + d_est_upper'*D*d_est_lower;
            L21 = A_new'/2 - X*(x_upper(:, 1) + x_lower(:,1))/2 - Cy_bold'*Y*y_tilde;
            L22 = X + Cy_bold'*Y*Cy_bold;
            L31 = - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
            L32 = Dyd_bold'*Y*Cy_bold;
            L33 = D + Dyd_bold'*Y*Dyd_bold;
            L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
            maximize(g)
            subject to
            L >= 0
            cvx_end
            x_lower(i, 1) = max(x_lower(i,1), g);
        end
    else
        % For x1 and so on
        k_start = max(1, k - Ne);
        N_e = k - k_start;
        y_est = y(:, k_start:k);
        y_est = y_est(:);
        u_est = u_out(:, k_start:k-1);
        u_est = u_est(:);
        if(Estimation.len ~= N_e) % Only update when estimation length is different
            Estimation.len = N_e;
            [Estimation.A_bold, Estimation.B_bold, Estimation.Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N_e);
            [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, N_e);
        end
        y_tilde = y_est - Dy_bold*u_est;
        d_est_lower = kron(ones(N_e + 1,1), d_lower);
        d_est_upper = kron(ones(N_e + 1,1), d_upper);
        % For each entry
        for i = 1:nx
            A_new  = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.A_bold;
            B_new  = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.B_bold;
            Bd_new = e(i, :) * [zeros(nx, nx*(N_e - 1)) eye(nx)] * Estimation.Bd_bold;
            % Upper bounds
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variable g(1);
            variable X(nx, nx) diagonal semidefinite;
            variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
            variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
            % LMI
            L11 = g-B_new*u_est+y_tilde'*Y*y_tilde+x_upper(:,k_start)'*X*x_lower(:,k_start)+d_est_upper'*D*d_est_lower;
            L21 = -A_new'/2 - X*(x_upper(:, k_start) + x_lower(:,k_start))/2 - Cy_bold'*Y*y_tilde;
            L22 = X + Cy_bold'*Y*Cy_bold;
            L31 = -Bd_new'/2 - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
            L32 = Dyd_bold'*Y*Cy_bold;
            L33 = D + Dyd_bold'*Y*Dyd_bold;
            L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
            minimize(g)
            subject to
            L >= 0
            cvx_end
            x_upper(i, k) = min(x_upper(i,k), g);
            % Lower bounds
            cvx_begin sdp quiet
            cvx_precision high
            % CVX variables
            variable g(1);
            variable X(nx, nx) diagonal semidefinite;
            variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
            variable D(size(d_est_lower, 1), size(d_est_lower, 1)) diagonal semidefinite;
            % LMI
            L11 = -g + B_new*u_est + y_tilde'*Y*y_tilde + x_upper(:,k_start)'*X*x_lower(:,k_start) + d_est_upper'*D*d_est_lower;
            L21 = A_new'/2 - X*(x_upper(:, k_start) + x_lower(:,k_start))/2 - Cy_bold'*Y*y_tilde;
            L22 = X + Cy_bold'*Y*Cy_bold;
            L31 = Bd_new'/2 - D*(d_est_lower+d_est_upper)/2 - Dyd_bold'*Y*y_tilde;
            L32 = Dyd_bold'*Y*Cy_bold;
            L33 = D + Dyd_bold'*Y*Dyd_bold;
            L = [L11 L21' L31'; L21 L22 L32'; L31 L32 L33];
            maximize(g)
            subject to
            L >= 0
            cvx_end
            x_lower(i, k) = max(x_lower(i,k), g);
        end
    end
    %% RMPC
    x_over = x_upper(:,k);
    x_under = x_lower(:,k);
    cvx_begin sdp quiet
    cvx_precision high
    % CVX variables
    variable g(1) nonnegative;
    variable m(1);
    variable u(Nc*nu ,1);
    variable X(nx, nx) diagonal semidefinite;
    variable X_check(nx, nx) diagonal semidefinite;
    variable D(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
    variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
    variable M(size(Cf_bold, 1), size(Cf_bold, 1)) diagonal;
    % LMIs
    L11 = Dz_bold'*Dz_bold;
    L21 = u'*(Dz_bold'*Dz_bold);
    L22 = g - z_bold_bar'*z_bold_bar + 2*z_bold_bar'*Dz_bold*u + x_over'*X*x_under+d_pred_upper'*D*d_pred_lower;
    L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
    L32 = Cz_bold'*(z_bold_bar - Dz_bold*u) - X*(x_over+x_under)/2;
    L33 = X - Cz_bold'*Cz_bold;
    L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
    L42 = Dzd_bold'*(z_bold_bar - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2;
    L43 = -Dzd_bold'*Cz_bold;
    L44 = D - Dzd_bold'*Dzd_bold;
    L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
    L_11 = 2*m + x_over'*X_check*x_under + d_pred_upper'*D_check*d_pred_lower;
    L_21 = f_bar - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
    L_22 = 2*M;
    L_31 = -X_check*(x_over+x_under)/2;
    L_32 = -Cf_bold';
    L_33 = X_check;
    L_41 = -D_check*(d_pred_lower+d_pred_upper)/2;
    L_42 = -Dfd_bold';
    L_43 = zeros(size(D_check, 1), size(X_check, 2));
    L_44 = D_check;
    L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
    minimize(g)
    subject to
    L >= 0
    L_ >= 0
    cvx_end
    u0 = u(1:nu);
    z = Cz * x(:,k) + Dz * u0 + Dzd * dk;
    cost(1, k) = z'*z;
    x(:, k+1) = A*x(:,k) + B*u0 + Bd*dk;
    u_out(:,k) = u0; 
end
%% Graphs
% Cost
figure(1);
plot(1:N_seq, cost, '*-');grid on
xlabel('Iterations');
title('Costs');xlim([1 N_seq]);
% Bounds
figure(2);
subplot(2,2,1); title('x1');hold on 
plot(1:N_seq, x_upper(1,1:N_seq), 'Color','red', 'LineStyle','-.');grid on
plot(1:N_seq, x(1,1:N_seq), 'Color','blue', 'Marker','*');
plot(1:N_seq, x_lower(1,1:N_seq), 'Color','black', 'LineStyle','-.');legend('upper', 'actual', 'lower');hold off
xlim([1 N_seq]);xlabel('Iterations');
subplot(2,2,2); title('x2');hold on 
plot(1:N_seq, x_upper(2,1:N_seq), 'Color','red', 'LineStyle','-.');grid on
plot(1:N_seq, x(2,1:N_seq), 'Color','blue', 'Marker','*');
plot(1:N_seq, x_lower(2,1:N_seq), 'Color','black', 'LineStyle','-.');legend('upper', 'actual', 'lower');hold off
xlim([1 N_seq]);xlabel('Iterations');
subplot(2,2,3); title('x3');hold on 
plot(1:N_seq, x_upper(3,1:N_seq), 'Color','red', 'LineStyle','-.');grid on
plot(1:N_seq, x(3,1:N_seq), 'Color','blue', 'Marker','*');
plot(1:N_seq, x_lower(3,1:N_seq), 'Color','black', 'LineStyle','-.');legend('upper', 'actual', 'lower');hold off
xlim([1 N_seq]);xlabel('Iterations');
subplot(2,2,4); title('x4');hold on 
plot(1:N_seq, x_upper(4,1:N_seq), 'Color','red', 'LineStyle','-.');grid on
plot(1:N_seq, x(4,1:N_seq), 'Color','blue', 'Marker','*');
plot(1:N_seq, x_lower(4,1:N_seq), 'Color','black', 'LineStyle','-.');legend('upper', 'actual', 'lower');hold off
xlim([1 N_seq]);xlabel('Iterations');