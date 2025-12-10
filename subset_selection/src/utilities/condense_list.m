function class_count = condense_list(annotations)

  class_count = zeros(1,17);

  for i = 1:length(annotations)
    switch (annotations{i})
      case "N"
        class_count(1) = class_count(1) + 1;
      case "A"
        class_count(2) = class_count(2) + 1;
      case "V"
        class_count(3) = class_count(3) + 1;
      case "Q"
        class_count(4) = class_count(4) + 1;
      case "/"
        class_count(5) = class_count(5) + 1;
      case "f"
        class_count(6) = class_count(6) + 1;
      case "F"
        class_count(7) = class_count(7) + 1;
      case "j"
        class_count(8) = class_count(8) + 1;
      case "L"
        class_count(9) = class_count(9) + 1;
      case "a"
        class_count(10) = class_count(10) + 1;
      case "J"
        class_count(11) = class_count(11) + 1;
      case "R"
        class_count(12) = class_count(12) + 1;
      case "!"
        class_count(13) = class_count(13) + 1;
      case "E"
        class_count(14) = class_count(14) + 1;
      case "s"
        class_count(15) = class_count(15) + 1;
      case "e"
        class_count(16) = class_count(16) + 1;
      otherwise
        class_count(17) = class_count(17) + 1;
    end
  end
end

