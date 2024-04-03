package com.vermeg.risk.repositories;

import com.vermeg.risk.entities.Credit;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PredictionRepository extends JpaRepository<Credit, Long> {

}
