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

xh = 2000* ones(nx,1);xl = -xh;
uh = 1000* ones(nu,1);ul = -uh;
xh_N = xh;xl_N = -xh_N;
fk = [xh;uh;-xl;-ul];
f_bar = [kron(ones(Nc, 1), fk);xh_N;-xl_N];
d = rand(nd, N_seq)*d_range_half*2 - d_range_half;

%% System output and input
% Output sequence
y = zeros(ny, N_seq);
% Input sequence
u_out = zeros(nu, N_seq);
% Cost Sequence
cost = zeros(1, N_seq);

%% Stacked System Implementation
[A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Nc);
[Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
[Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
z_bold_bar = zeros(size(Cz_bold, 1), 1);

%% Iterations with method B
x = zeros(nx, N_seq);
x(:,1) = x0;
x_lower = -5 *ones(nx, 1);
x_upper = 5* ones(nx, 1);
for k = 1:N_seq
    dk = d(:,k);
    y(:, k) = Cy * x(:,k) + Dyd * dk;
    k_start = max(1, k - Ne);
    N_e = k - k_start;
    y_est = y(:, k_start:k);
    y_est = y_est(:);
    d_pred_lower = kron(ones(Nc+N_e+1, 1), d_lower);
    d_pred_upper = kron(ones(Nc+N_e+1, 1), d_upper);
    if(N_e ~= Ne)
        %% Stacked System Implementation
        [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
        [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
        if(N_e == 0)
            y_tilde = y_est;
            Cy_bold = Cy;
            Dyd_bold = Dyd;
            f0 = f_bar;
            z0 = z_bold_bar;
        else
            [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, N_e);
            [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, N_e);
            u_est = u_out(:, k_start:k-1);
            u_est = u_est(:);
            y_tilde = y_est - Dy_bold*u_est;
            f0 = f_bar - Cf_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
            z0 = z_bold_bar - Cz_bold * [zeros(nx, nx*(N_e - 1)) eye(nx)] * B_bold *u_est;
            tmp = [zeros(nx, nx*(N_e - 1)) eye(nx)] * Bd_bold;
            tmp = tmp(:, 1:end - nd);
            Dfd_bold = [Cf_bold*tmp Dfd_bold];
            Dzd_bold = [Cz_bold*tmp Dzd_bold];
            Cf_bold = Cf_bold * A^N_e;
            Cz_bold = Cz_bold * A^N_e;
        end
        Dyd_bold = [Dyd_bold zeros(size(Dyd_bold, 1), nd*Nc)];
        % LMI
        cvx_begin sdp quiet
        cvx_precision high
        % CVX variables
        variable g(1) nonnegative;
        variable m(1);
        variable u(Nc*nu ,1);
        variable X(nx, nx) diagonal semidefinite;
        variable X_check(nx, nx) diagonal semidefinite;
        variable Y(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
        variable Y_check(size(y_tilde, 1), size(y_tilde, 1)) diagonal semidefinite;
        variable D(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
        variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1)) diagonal semidefinite;
        variable M(size(Cf_bold, 1), size(Cf_bold, 1)) diagonal;
        % LMIs
        L11 = Dz_bold'*Dz_bold;
        L21 = u'*(Dz_bold'*Dz_bold);
        L22 = g - z0'*z0 + 2*z0'*Dz_bold*u + y_tilde'*Y*y_tilde + d_pred_upper'*D*d_pred_lower + x_upper'*X*x_lower;
        L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
        L32 = Cz_bold'*(z0 - Dz_bold*u) - X*(x_lower+x_upper)/2 - Cy_bold'*Y*y_tilde;
        L33 = X - Cz_bold'*Cz_bold + Cy_bold'*Y*Cy_bold;
        L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
        L42 = Dzd_bold'*(z0 - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2 - Dyd_bold'*Y*y_tilde;
        L43 = -Dzd_bold'*Cz_bold + Dyd_bold'*Y*Cy_bold;
        L44 = D - Dzd_bold'*Dzd_bold + Dyd_bold'*Y*Dyd_bold;
        L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
        L_11 = 2*m + y_tilde'*Y_check*y_tilde + d_pred_upper'*D_check*d_pred_lower + x_upper'*X_check*x_lower;
        L_21 = f0 - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
        L_22 = 2*M;
        L_31 = -Cy_bold'*Y_check*y_tilde - X_check*(x_lower+x_upper)/2;
        L_32 = -Cf_bold';
        L_33 = X_check - Cy_bold'*Y_check*Cy_bold;
        L_41 = -Dyd_bold'*Y_check*y_tilde - D_check*(d_pred_lower+d_pred_upper)/2;
        L_42 = -Dfd_bold';
        L_43 = Dyd_bold'*Y_check*Cy_bold;
        L_44 = D_check;
        L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
        minimize(g)
        subject to
        L >= 0
        L_ >= 0
        cvx_end
    else
        u_est = u_out(:, k_start:k-1);
        u_est = u_est(:);
        % estimation horizon is long enough
        if(N_e == Ne)
            [Cf_bold, Df_bold, Dfd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cf, Cf_hat, Df, Dfd, Cf_hat*Bd.*0, Nc);
            [Cz_bold, Dz_bold, Dzd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cz, Cz_hat, Dz, Dzd, Cz_hat*Bd.*0, Nc);
            [A_bold, B_bold, Bd_bold] = compute_A_and_B_and_Bd(A, B, Bd, Ne);
            [Cy_bold, Dy_bold, Dyd_bold] = compute_C_and_D_and_Dd(A, B, Bd, Cy, Cy, Dy, Dyd, Dyd, Ne);
            f0 = f_bar - Cf_bold * [zeros(nx, nx*(Ne - 1)) eye(nx)] * B_bold *u_est;
            z0 = z_bold_bar - Cz_bold * [zeros(nx, nx*(Ne - 1)) eye(nx)] * B_bold *u_est;
            tmp = [zeros(nx, nx*(Ne - 1)) eye(nx)] * Bd_bold;
            tmp = tmp(:, 1:end - nd);
            Dfd_bold = [Cf_bold*tmp Dfd_bold];
            Dzd_bold = [Cz_bold*tmp Dzd_bold];
            Cf_bold = Cf_bold * A^Ne;
            Cz_bold = Cz_bold * A^Ne;
            Dyd_bold = [Dyd_bold zeros(size(Dyd_bold, 1), nd*Nc)];
        end
        y_tilde = y_est - Dy_bold*u_est;
        % LMI
        cvx_begin sdp quiet
        cvx_precision high
        % CVX variables
        variable g(1) nonnegative;
        variable m(1);
        variable u(Nc*nu ,1);
        variable Y(size(y_tilde, 1), size(y_tilde, 1))  diagonal semidefinite;
        variable Y_check(size(y_tilde, 1), size(y_tilde, 1))  diagonal semidefinite;
        variable D(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
        variable D_check(size(d_pred_lower, 1),size(d_pred_lower, 1))  diagonal semidefinite;
        variable M(size(Cf_bold, 1), size(Cf_bold, 1)) diagonal;
        % LMIs
        L11 = Dz_bold'*Dz_bold;
        L21 = u'*(Dz_bold'*Dz_bold);
        L22 = g - z0'*z0 + 2*z0'*Dz_bold*u + y_tilde'*Y*y_tilde + d_pred_upper'*D*d_pred_lower;
        L31 = zeros(size(Cz_bold, 2), size(Dz_bold, 2));
        L32 = Cz_bold'*(z0 - Dz_bold*u) - Cy_bold'*Y*y_tilde;
        L33 = -Cz_bold'*Cz_bold + Cy_bold'*Y*Cy_bold;
        L41 = zeros(size(Dzd_bold, 2), size(Dz_bold, 2));
        L42 = Dzd_bold'*(z0 - Dz_bold*u) - D*(d_pred_lower+d_pred_upper)/2 - Dyd_bold'*Y*y_tilde;
        L43 = -Dzd_bold'*Cz_bold + Dyd_bold'*Y*Cy_bold;
        L44 = D - Dzd_bold'*Dzd_bold + Dyd_bold'*Y*Dyd_bold;
        L = [L11 L21' L31' L41'; L21 L22 L32' L42'; L31 L32 L33 L43'; L41 L42 L43 L44];
        L_11 = 2*m + y_tilde'*Y_check*y_tilde + d_pred_upper'*D_check*d_pred_lower;
        L_21 = f0 - Df_bold*u - M*ones(size(Cf_bold, 1), 1) - ones(size(Cf_bold, 1), 1)*m;
        L_22 = 2*M;
        L_31 = -Cy_bold'*Y_check*y_tilde;
        L_32 = -Cf_bold';
        L_33 = Cy_bold'*Y_check*Cy_bold;
        L_41 = -Dyd_bold'*Y_check*y_tilde-D_check*(d_pred_lower+d_pred_upper)/2;
        L_42 = -Dfd_bold';
        L_43 = Dyd_bold'*Y_check*Cy_bold;
        L_44 = D_check;
        L_ = [L_11 L_21' L_31' L_41'; L_21 L_22 L_32' L_42'; L_31 L_32 L_33 L_43'; L_41 L_42 L_43 L_44];
        minimize(g)
        subject to
        L >= 0
        L_ >= 0
        cvx_end
    end
    u0 = u(1:nu);
    z = Cz * x(:,k) + Dz * u0 + Dzd * dk;
    cost(1, k) = z'*z;
    x(:, k+1) = A*x(:,k) + B*u0 + Bd*dk;
    u_out(:,k) = u0;
end
%% Graphs
% Cost
figure(1);
plot(1:N_seq, cost,'Color','blue','Marker', '*');
xlabel('Iterations');title('Costs');xlim([1 N_seq]);grid on
% States
figure(2);
x_up = ceil(max(x(:)));
x_down = floor(min(x(:)));
subplot(2,2,1); 
plot(1:N_seq, x(1,1:N_seq), 'Color','blue', 'Marker','*');
title('x1');xlim([1 N_seq]);ylim([x_down x_up]);xlabel('Iterations');grid on
subplot(2,2,2); 
plot(1:N_seq, x(2,1:N_seq), 'Color','blue', 'Marker','*');
title('x2');xlim([1 N_seq]);ylim([x_down x_up]);xlabel('Iterations');grid on
subplot(2,2,3); 
plot(1:N_seq, x(3,1:N_seq), 'Color','blue', 'Marker','*');
title('x3');xlim([1 N_seq]);ylim([x_down x_up]);xlabel('Iterations');grid on
subplot(2,2,4); 
plot(1:N_seq, x(4,1:N_seq), 'Color','blue', 'Marker','*');
title('x4');xlim([1 N_seq]);ylim([x_down x_up]);xlabel('Iterations');grid on