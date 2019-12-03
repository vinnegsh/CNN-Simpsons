library(keras)
library(reticulate)
library(OpenImageR)
library(tidyverse)
library(jpeg)
library(ramify)
library(caret)
library(shiny)
library(shinythemes)

pic_size <- 64
batch_size <- 32

train_directory <- "simpsons_dataset/train"
test_directory <- "simpsons_dataset/test"

train_generator <- flow_images_from_directory(train_directory, generator = image_data_generator(rescale=1./255,
                                                                                                rotation_range = 10,
                                                                                                width_shift_range=0.1,
                                                                                                height_shift_range=0.1,
                                                                                                horizontal_flip=T), 
                                              target_size = c(pic_size, pic_size), color_mode = "rgb",
                                              class_mode = "categorical", batch_size = batch_size, shuffle = FALSE,
                                              seed = 188228)

validation_generator <- flow_images_from_directory(test_directory, generator = image_data_generator(rescale=1./255),
                                                   target_size = c(pic_size, pic_size), color_mode = "rgb", classes = NULL,
                                                   class_mode = "categorical", batch_size = batch_size, shuffle = TRUE,
                                                   seed = 188228)
train_samples = 16146
validation_samples = 2848

modelo <- keras_model_sequential()
modelo %>% 
  layer_conv_2d(filters=32,
                kernel_size = c(3,3),
                padding = "same",
                input_shape = c(pic_size,pic_size,3),
                activation = "relu") %>%
  layer_conv_2d(filters=32,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filters=64,
                kernel_size = c(3,3),
                padding="same",
                activation = "relu") %>%
  layer_conv_2d(filters=64,
                kernel_size=c(3,3),
                activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filters=256,
                kernel_size=c(3,3),
                padding="same",
                activation = "relu") %>%
  layer_conv_2d(filters=256, 
                kernel_size=c(3,3),
                activation="relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.2) %>%
  
  layer_flatten() %>%
  layer_dense(units=1024, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units=18, activation = "softmax")%>%
  
  compile(loss="categorical_crossentropy", optimizer=optimizer_sgd(lr=0.01,
                                                                   decay=1e-6,
                                                                   momentum = 0.9,
                                                                   nesterov = TRUE),
          metrics=c("accuracy"))

hist <- modelo %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 60, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)  

lista <- tribble(~num, ~pers,
                 0,"Abraham Grampa Simpson",
                 1,"Apu Nahasapeemapetilon",
                 2,"Bart Simpson",
                 3,"Charles Montgomery Burns",
                 4,"Chief Wiggum",
                 5,"Comic Book Guy",
                 6,"Edna Krabappel",
                 7,"Homer Simpson",
                 8,"Kent Brockman",
                 9,"Krusty The Clown",
                 10,"Lisa Simpson",
                 11,"Marge Simpson",
                 12,"Milhouse van Houten",
                 13,"Moe Szylak",
                 14,"Ned Flanders",
                 15,"Nelson Muntz",
                 16, "Principal Skinner",
                 17, "Sideshow Bob")

nomes <- lista$pers

pred <- function(url, personagem_certo=boneco){
  tryCatch({
    if (url==''){return("")}
    z <- tempfile()
    download.file(url, z, mode="wb")
    tryCatch({
      img <- readJPEG(z) %>% 
        resize(64,64) %>% array_reshape(c(1,64,64,3))
      file.remove(z)
      personagem <- modelo %>% predict_classes(img)
      personagem_correto <- lista$num[which(lista$pers == personagem_certo)]
      prob <- modelo %>% predict_proba(img)
      lista_resp <- list(lista$pers[personagem+1], round(max(prob),3), round(prob[[personagem_correto+1]],3), prob)
      
      return(lista_resp)
    }, error=function(e) {
      return(list("ERRO", 0, 0))
    }, warning=function(w) {
      return(list("ERRO", 0, 0))
    })
  }, error=function(e) {
    return(list("ERRO", 0, 0))
  }, warning=function(w) {
    return(list("ERRO", 0, 0))
  })
}

ui <- fillPage(
  theme=shinytheme("slate"),
  titlePanel("Projeto Final ME906 - CNN para classificação de personagens de Os Simpsons"),
  fluidRow(
    column(12,align="center", wellPanel(
      textInput("my_url", "Link da Imagem", value='', width='100%'),
      selectInput("select", label=("Qual o Personagem?"), choices=nomes),
      actionButton("limpar", "Limpar"),
      actionButton("predict", "Adivinhar o Personagem")
    ))),
  fluidRow(
    column(6,align="center", wellPanel(
      htmlOutput("picture"),
      textOutput("pred"),
      tags$head(tags$style("#pred{font-size: 16px;}"))
    )),
    column(6, align="center", wellPanel(
      textOutput("acertos"),
      textOutput("erros"),
      textOutput("media")
    ))
  )
  
  
)


server <- function(input, output, session) {
  observeEvent(input$limpar,{
    output$my_url <- renderText({ })
    output$pred <- renderText({ })
    output$picture <- renderText({ })
  })
  
  observeEvent(input$select,
               {
                 boneco <<- input$select
               })
  
  
  
  acertos <- reactiveValues(countervalue=0)
  erros <- reactiveValues(countervalue=0)
  observeEvent(input$predict,{
    if(pred(input$my_url)[[1]] !="ERRO"){
      output$picture <-
        renderText({
          c(
            '<img src="',
            input$my_url,
            '", width=350px, height=250px>'
          )
        })
    }
    else{
      output$picture <- renderText({ })
    }
    
    #output$pred <- renderText({paste("O modelo acha que o personagem é ",pred(input$my_url)[[1]], " com P(X)=", round(max(pred(input$my_url)[[2]]),2), sep="")})
    if(pred(input$my_url)[[1]] !="ERRO"){
      if(pred(input$my_url)[[1]] == input$select){
        acertos$countervalue <- acertos$countervalue + 1
        output$pred <- renderText({paste("O modelo acha que o personagem é ",pred(input$my_url)[[1]], " com P(X)=", pred(input$my_url)[[2]],".",sep="")})
      }
      else{
        erros$countervalue <- erros$countervalue + 1
        output$pred <- renderText({paste("O modelo acha que o personagem é ",pred(input$my_url)[[1]], " com P(X)=", pred(input$my_url)[[2]],
                                         " e o modelo acha que é o personagem ", input$select ," com P(X)=", pred(input$my_url)[[3]],".", sep="")})
      }
    }
    else{
      output$pred <- renderText({"Insira uma imagem .jpg válida"})
    }
    
    output$acertos <- renderText({paste("Acertos: ",acertos$countervalue, sep="")})
    output$erros <- renderText({paste("Erros: ",erros$countervalue, sep="")})
    output$media <- renderText({paste("Por enquanto, o modelo acertou ",
                                      round((acertos$countervalue / (acertos$countervalue + erros$countervalue))*100,2), "% das vezes.", sep="")})})
  
  
}

shinyApp(ui = ui, server = server)
