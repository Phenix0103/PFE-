package com.vermeg.risk.repositories;

import com.vermeg.risk.entities.Question;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuestionRepository extends JpaRepository<Question, Long> {
}
