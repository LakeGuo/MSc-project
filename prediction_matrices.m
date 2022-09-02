function [Phi, Gamma, Lambda] = prediction_matrices(A, B_d, B_u, N)
    Phi = [];
    for i = 1:N
        Phi = [Phi; A^i];
    end
    %% B_bold
    Gamma = kron(eye(N), B_u);
    for i = 1:N-1
        Gamma = Gamma + kron(diag(ones(N-i, 1), -i), (A^i)*B_u);
    end
    %% Bd_bold
    Lambda = kron(eye(N), B_d);
    for i = 1:N-1
        Lambda = Lambda + kron(diag(ones(N-i, 1), -i), (A^i)*B_d);
    end
end