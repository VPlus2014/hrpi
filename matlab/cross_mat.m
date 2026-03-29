function mat = cross_mat(vect)
mat = [...
    0 -vect(3) vect(2);
    vect(3) 0 -vect(1);
    -vect(2) vect(1) 0];
end

