# --- 1. Install Shiny ---
# Run this in your console one time:
# install.packages("shiny")

# --- 2. Create your 'app.R' file with this code ---

library(shiny)
library(dplyr)
library(ggplot2)

# Define the User Interface (the "look" of the webpage)
ui <- fluidPage(
    
    # App title
    titlePanel("Live Classroom Attentiveness Dashboard"),
    
    # A row for the summary boxes
    fluidRow(
        column(4, wellPanel(h3("Total Students"), textOutput("totalStudents"))),
        column(4, wellPanel(h3("Class Average Score"), textOutput("classAvg"))),
        column(4, wellPanel(h3("At-Risk Students"), textOutput("atRiskCount")))
    ),
    
    hr(), # A horizontal line
    
    # A row for the two main plots
    fluidRow(
        column(6, plotOutput("lineGraph")), # 6 of 12 columns
        column(6, plotOutput("boxPlot"))    # 6 of 12 columns
    ),
    
    hr(), # A horizontal line
    
    # A row for the summary table
    fluidRow(
        column(12, dataTableOutput("summaryTable"))
    )
)

# Define the Server (the "brain" that builds the plots/tables)
server <- function(input, output) {

    # --- 3. Reactive Data Loading ---
    # This function will re-run if the data file changes
    # It makes the dashboard "live"
    loadData <- reactive({
        # Invalidate data every 10 seconds to check for new CSV data
        invalidateLater(10000) 
        
        data <- read.csv("classroom_log.csv")
        data$status <- as.factor(data$status)
        data$timestamp <- as.POSIXct(data$timestamp, format = "%Y-%m-%dT%H:%M:%OS")
        return(data)
    })

    # --- 4. Render the Plots and Tables ---
    # We just copy/paste our RMarkdown logic here
    
    output$totalStudents <- renderText({
        data <- loadData()
        length(unique(data$roll_no))
    })
    
    output$classAvg <- renderText({
        data <- loadData()
        paste0(round(mean(data$attentiveness_score) * 100, 1), "%")
    })
    
    output$atRiskCount <- renderText({
        data <- loadData()
        data %>%
          group_by(roll_no) %>%
          summarise(avg_score = mean(attentiveness_score)) %>%
          filter(avg_score < 0.6) %>%
          nrow()
    })

    output$lineGraph <- renderPlot({
        data <- loadData()
        data %>%
          mutate(time_bin = cut(timestamp, breaks = "20 sec")) %>%
          group_by(time_bin) %>%
          summarise(class_avg_score = mean(attentiveness_score)) %>%
          mutate(time_bin = as.POSIXct(time_bin)) %>%
          
          ggplot(aes(x = time_bin, y = class_avg_score)) +
          geom_line(color = "#0072B2", size = 1.2) +
          geom_point(color = "#0072B2", size = 2) +
          labs(title = "Average Class Attentiveness Over Time", x = "Session Time", y = "Avg. Score") +
          ylim(0, 1) + theme_minimal(base_size = 14)
    })
    
    output$boxPlot <- renderPlot({
        data <- loadData()
        ggplot(data, aes(x = reorder(roll_no, attentiveness_score), y = attentiveness_score, fill = roll_no)) +
          geom_boxplot() +
          labs(title = "Student Attentiveness Distribution", x = "Student", y = "Score") +
          theme_minimal(base_size = 14) +
          theme(legend.position = "none") +
          coord_flip() # Horizontal is easier to read
    })
    
    output$summaryTable <- renderDataTable({
        data <- loadData()
        data %>%
          group_by(roll_no) %>%
          summarise(
            avg_score = round(mean(attentiveness_score), 2),
            median_score = round(median(attentiveness_score), 2),
            std_dev = round(sd(attentiveness_score), 2),
            time_points = n()
          ) %>%
          arrange(avg_score)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)