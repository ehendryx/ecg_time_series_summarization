function class_count = condense_letter(ls)

  class_count = zeros(1,27);

  len = length(ls);

  for i = 1:len
    switch(ls{i})
      case "A"
        class_count(1) = class_count(1) + 1;
      case "B"
        class_count(2) = class_count(2) + 1;
      case "C"
        class_count(3) = class_count(3) + 1;
      case "D"
        class_count(4) = class_count(4) + 1;
      case "E"
        class_count(5) = class_count(5) + 1;
      case "F"
        class_count(6) = class_count(6) + 1;
      case "G"
        class_count(7) = class_count(7) + 1;
      case "H"
        class_count(8) = class_count(8) + 1;
      case "I"
        class_count(9) = class_count(9) + 1;
      case "J"
        class_count(10) = class_count(10) + 1;
      case "K"
        class_count(11) = class_count(11) + 1;
      case "L"
        class_count(12) = class_count(12) + 1;
      case "M"
        class_count(13) = class_count(13) + 1;
      case "N"
        class_count(14) = class_count(14) + 1;
      case "O"
        class_count(15) = class_count(15) + 1;
      case "P"
        class_count(16) = class_count(16) + 1;
      case "Q"
        class_count(17) = class_count(17) + 1;
      case "R"
        class_count(18) = class_count(18) + 1;
      case "S"
        class_count(19) = class_count(19) + 1;
      case "T"
        class_count(20) = class_count(20) + 1;
      case "U"
        class_count(21) = class_count(21) + 1;
      case "V"
        class_count(22) = class_count(22) + 1;
      case "W"
        class_count(23) = class_count(23) + 1;
      case "X"
        class_count(24) = class_count(24) + 1;
      case "Y"
        class_count(25) = class_count(25) + 1;
      case "Z"
        class_count(26) = class_count(26) + 1;
      otherwise
        class_count(27) = class_count(27) + 1;
  end
end

